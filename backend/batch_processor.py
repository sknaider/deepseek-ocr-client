#!/usr/bin/env python3
"""
DeepSeek-OCR Batch Processor
Processes folders of medical documents with GPU optimization and error recovery

Optimized for RTX 5090 32GB VRAM with medical document processing in Peru
"""

import os
import sys
import json
import time
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import traceback

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image

# GPU memory tracking
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU memory tracking will be limited.")


@dataclass
class ProcessingResult:
    """Result of processing a single document"""
    filepath: str
    filename: str
    success: bool
    error_message: Optional[str]
    processing_time_seconds: float
    output_dir: str
    markdown_path: Optional[str]
    metadata_path: Optional[str]
    boxes_image_path: Optional[str]
    document_type: Optional[str]
    char_count: int
    word_count: int
    confidence_score: float
    gpu_memory_used_mb: float
    timestamp_utc: str
    model_used: str
    prompt_type: str
    image_dimensions: Optional[Tuple[int, int]]
    file_size_bytes: int


@dataclass
class BatchMetrics:
    """Metrics for entire batch processing"""
    total_files: int
    successful: int
    failed: int
    skipped: int
    total_time_seconds: float
    average_time_per_file: float
    throughput_files_per_minute: float
    peak_gpu_memory_mb: float
    average_gpu_memory_mb: float
    total_chars_extracted: int
    total_words_extracted: int
    error_rate_percent: float
    start_time: str
    end_time: str
    failed_files: List[str]


class GPUMemoryMonitor:
    """Monitor GPU memory usage for RTX 5090 optimization"""

    def __init__(self):
        self.current_usage_mb = 0.0
        self.peak_usage_mb = 0.0
        self.available_mb = 0.0
        self.total_mb = 0.0
        self.pynvml_initialized = False

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.pynvml_initialized = True
            except Exception as e:
                print(f"Warning: Could not initialize pynvml: {e}")

    def update(self) -> float:
        """Update and return current GPU memory usage in MB"""
        if torch.cuda.is_available():
            # PyTorch method (always available)
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
            self.current_usage_mb = reserved  # Use reserved as it's more accurate

            # Update peak
            if self.current_usage_mb > self.peak_usage_mb:
                self.peak_usage_mb = self.current_usage_mb

            # Try to get total memory with pynvml (more accurate)
            if self.pynvml_initialized:
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    self.total_mb = info.total / (1024 ** 2)
                    self.available_mb = info.free / (1024 ** 2)
                except:
                    pass

        return self.current_usage_mb

    def get_usage_percent(self) -> float:
        """Get current GPU memory usage as percentage"""
        if self.total_mb > 0:
            return (self.current_usage_mb / self.total_mb) * 100
        return 0.0

    def clear_cache(self):
        """Clear CUDA cache to free fragmented memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def __del__(self):
        if self.pynvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


class BatchProcessor:
    """
    Batch processor for DeepSeek-OCR with medical document optimization

    Features:
    - Recursive folder processing
    - Concurrent GPU processing
    - GPU memory monitoring and auto-adjustment
    - Error recovery with retries
    - Progress tracking
    - Metadata generation
    - Medical document presets
    """

    def __init__(self, config_path: str = "batch_config.yaml"):
        """
        Initialize batch processor

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.model = None
        self.tokenizer = None
        self.gpu_monitor = GPUMemoryMonitor()
        self.processing_lock = Lock()
        self.results: List[ProcessingResult] = []
        self.failed_files: List[Tuple[str, str]] = []  # (filepath, error)

        # Statistics
        self.stats = {
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'total_chars': 0,
            'total_words': 0,
            'gpu_memory_samples': []
        }

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def _setup_logging(self) -> logging.Logger:
        """Setup logging with file and console handlers"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_file = log_config.get('log_file', 'batch_processing.log')
        log_format = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')

        # Create logger
        logger = logging.getLogger('BatchProcessor')
        logger.setLevel(log_level)
        logger.handlers.clear()  # Remove existing handlers

        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

        # Console handler if enabled
        if log_config.get('console_output', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(console_handler)

        return logger

    def load_model(self):
        """Load DeepSeek-OCR model and tokenizer with GPU optimization"""
        if self.model is not None and self.tokenizer is not None:
            self.logger.info("Model already loaded")
            return

        model_config = self.config['model']
        model_name = model_config['model_name']
        cache_dir = model_config['cache_dir']

        self.logger.info(f"Loading model: {model_name}")
        self.logger.info(f"Cache directory: {cache_dir}")

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Check GPU availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            self.logger.info(f"GPU available: {gpu_name} ({total_memory:.1f} GB)")
        else:
            self.logger.warning("No GPU available! Processing will be very slow.")

        # Load tokenizer
        self.logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        # Load model
        self.logger.info("Loading model (this may take a few minutes)...")
        gpu_config = self.config.get('gpu', {})
        use_flash_attention = gpu_config.get('prefer_flash_attention', True)

        try:
            if use_flash_attention:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    _attn_implementation='flash_attention_2',
                    trust_remote_code=True,
                    use_safetensors=True,
                    cache_dir=cache_dir
                )
                self.logger.info("Using flash attention 2 (optimized)")
            else:
                raise Exception("Flash attention disabled in config")
        except Exception as e:
            self.logger.warning(f"Flash attention not available: {e}")
            self.logger.info("Using default attention mechanism")
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_safetensors=True,
                cache_dir=cache_dir
            )

        # Move to GPU with bfloat16
        self.model = self.model.eval()
        if torch.cuda.is_available():
            use_bfloat16 = gpu_config.get('use_bfloat16', True)
            if use_bfloat16:
                self.model = self.model.cuda().to(torch.bfloat16)
                self.logger.info("Model loaded on GPU with bfloat16 precision")
            else:
                self.model = self.model.cuda()
                self.logger.info("Model loaded on GPU with float32 precision")
        else:
            self.logger.info("Model loaded on CPU")

        # Report initial GPU memory
        self.gpu_monitor.update()
        self.logger.info(f"Model loaded. GPU memory: {self.gpu_monitor.current_usage_mb:.1f} MB")

    def find_files(self, input_dir: str) -> List[str]:
        """
        Recursively find all supported image files in directory

        Args:
            input_dir: Input directory path

        Returns:
            List of file paths
        """
        supported_exts = self.config['general']['supported_extensions']
        max_size_mb = self.config['general']['max_file_size_mb']

        files = []
        self.logger.info(f"Scanning directory: {input_dir}")

        for root, dirs, filenames in os.walk(input_dir):
            for filename in filenames:
                # Check extension
                ext = os.path.splitext(filename)[1].lower()
                if ext not in supported_exts:
                    continue

                filepath = os.path.join(root, filename)

                # Check file size
                try:
                    file_size_mb = os.path.getsize(filepath) / (1024 ** 2)
                    if file_size_mb > max_size_mb:
                        self.logger.warning(f"Skipping {filename} (too large: {file_size_mb:.1f} MB)")
                        self.stats['skipped'] += 1
                        continue
                except OSError as e:
                    self.logger.warning(f"Cannot access {filepath}: {e}")
                    continue

                files.append(filepath)

        self.logger.info(f"Found {len(files)} files to process")
        return files

    def process_single_file(
        self,
        filepath: str,
        output_base_dir: str,
        input_base_dir: str,
        preset_name: str = "generic"
    ) -> ProcessingResult:
        """
        Process a single file with OCR

        Args:
            filepath: Path to input file
            output_base_dir: Base output directory
            input_base_dir: Base input directory (for preserving structure)
            preset_name: Medical preset name from config

        Returns:
            ProcessingResult object
        """
        start_time = time.time()
        filename = os.path.basename(filepath)

        self.logger.info(f"Processing: {filename}")

        # Get preset configuration
        presets = self.config.get('medical_presets', {})
        preset = presets.get(preset_name, presets.get('generic', {}))

        prompt_type = preset.get('prompt_type', self.config['model']['default_prompt_type'])
        base_size = preset.get('base_size', self.config['model']['base_size'])
        image_size = preset.get('image_size', self.config['model']['image_size'])
        crop_mode = self.config['model'].get('crop_mode', True)

        # Determine output directory
        if self.config['general']['preserve_folder_structure']:
            # Maintain folder hierarchy
            rel_path = os.path.relpath(os.path.dirname(filepath), input_base_dir)
            output_dir = os.path.join(output_base_dir, rel_path)
        else:
            output_dir = output_base_dir

        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectory for this file's outputs
        file_base = os.path.splitext(filename)[0]
        file_output_dir = os.path.join(output_dir, file_base)
        os.makedirs(file_output_dir, exist_ok=True)

        # Get file info
        try:
            file_size = os.path.getsize(filepath)
            img = Image.open(filepath)
            img_dimensions = img.size
            img.close()
        except Exception as e:
            return ProcessingResult(
                filepath=filepath,
                filename=filename,
                success=False,
                error_message=f"Cannot read file: {e}",
                processing_time_seconds=time.time() - start_time,
                output_dir=file_output_dir,
                markdown_path=None,
                metadata_path=None,
                boxes_image_path=None,
                document_type=preset_name,
                char_count=0,
                word_count=0,
                confidence_score=0.0,
                gpu_memory_used_mb=0.0,
                timestamp_utc=datetime.utcnow().isoformat(),
                model_used=self.config['model']['model_name'],
                prompt_type=prompt_type,
                image_dimensions=None,
                file_size_bytes=0
            )

        # Prepare prompt
        prompt_configs = {
            'document': '<image>\n<|grounding|>Convert the document to markdown. ',
            'ocr': '<image>\n<|grounding|>OCR this image. ',
            'free': '<image>\nFree OCR. ',
            'figure': '<image>\nParse the figure. ',
            'describe': '<image>\nDescribe this image in detail. '
        }
        prompt = prompt_configs.get(prompt_type, prompt_configs['document'])

        # Process with model
        try:
            # Monitor GPU memory before processing
            gpu_memory_before = self.gpu_monitor.update()

            # Perform inference
            self.logger.debug(f"Running OCR with prompt type: {prompt_type}")

            # Suppress stdout to avoid clutter (model prints a lot)
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

            try:
                self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=filepath,
                    output_path=file_output_dir,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=True,
                    test_compress=False  # Disable compression for speed
                )
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout

            # Monitor GPU memory after processing
            gpu_memory_after = self.gpu_monitor.update()
            gpu_memory_used = max(gpu_memory_after - gpu_memory_before, 0)

            # Find output files
            markdown_path = None
            boxes_image_path = None
            result_text = ""

            # Look for markdown/text output
            for ext in ['.mmd', '.md', '.txt']:
                potential_path = os.path.join(file_output_dir, f'result{ext}')
                if os.path.exists(potential_path):
                    markdown_path = potential_path
                    with open(markdown_path, 'r', encoding='utf-8') as f:
                        result_text = f.read()
                    break

            # Look for boxes image
            boxes_path = os.path.join(file_output_dir, 'result_with_boxes.jpg')
            if os.path.exists(boxes_path):
                boxes_image_path = boxes_path

            # Calculate metrics
            char_count = len(result_text)
            word_count = len(result_text.split())

            # Create metadata
            metadata = {
                'filename': filename,
                'filepath': filepath,
                'document_type': preset_name,
                'processing_time_seconds': time.time() - start_time,
                'model_used': self.config['model']['model_name'],
                'prompt_type': prompt_type,
                'image_dimensions': list(img_dimensions),
                'file_size_bytes': file_size,
                'confidence_score': 1.0,  # TODO: Extract from model output
                'gpu_memory_used_mb': gpu_memory_used,
                'timestamp_utc': datetime.utcnow().isoformat(),
                'char_count': char_count,
                'word_count': word_count,
                'base_size': base_size,
                'image_size': image_size,
                'crop_mode': crop_mode
            }

            # Save metadata JSON
            metadata_path = os.path.join(file_output_dir, 'metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            processing_time = time.time() - start_time
            self.logger.info(
                f"✓ {filename} - {processing_time:.1f}s - "
                f"{char_count} chars - {gpu_memory_used:.0f}MB GPU"
            )

            return ProcessingResult(
                filepath=filepath,
                filename=filename,
                success=True,
                error_message=None,
                processing_time_seconds=processing_time,
                output_dir=file_output_dir,
                markdown_path=markdown_path,
                metadata_path=metadata_path,
                boxes_image_path=boxes_image_path,
                document_type=preset_name,
                char_count=char_count,
                word_count=word_count,
                confidence_score=1.0,
                gpu_memory_used_mb=gpu_memory_used,
                timestamp_utc=metadata['timestamp_utc'],
                model_used=self.config['model']['model_name'],
                prompt_type=prompt_type,
                image_dimensions=img_dimensions,
                file_size_bytes=file_size
            )

        except Exception as e:
            error_msg = f"OCR failed: {str(e)}"
            self.logger.error(f"✗ {filename} - {error_msg}")
            self.logger.debug(traceback.format_exc())

            return ProcessingResult(
                filepath=filepath,
                filename=filename,
                success=False,
                error_message=error_msg,
                processing_time_seconds=time.time() - start_time,
                output_dir=file_output_dir,
                markdown_path=None,
                metadata_path=None,
                boxes_image_path=None,
                document_type=preset_name,
                char_count=0,
                word_count=0,
                confidence_score=0.0,
                gpu_memory_used_mb=self.gpu_monitor.current_usage_mb,
                timestamp_utc=datetime.utcnow().isoformat(),
                model_used=self.config['model']['model_name'],
                prompt_type=prompt_type,
                image_dimensions=img_dimensions if 'img_dimensions' in locals() else None,
                file_size_bytes=file_size if 'file_size' in locals() else 0
            )

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        preset: str = "generic"
    ) -> BatchMetrics:
        """
        Process entire folder of documents

        Args:
            input_dir: Input directory containing medical documents
            output_dir: Output directory for results
            preset: Medical preset name from config

        Returns:
            BatchMetrics object with processing statistics
        """
        batch_start = time.time()
        self.stats['start_time'] = datetime.utcnow().isoformat()

        self.logger.info("="*80)
        self.logger.info("BATCH PROCESSING STARTED")
        self.logger.info("="*80)
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Medical preset: {preset}")

        # Load model
        self.load_model()

        # Find all files
        files = self.find_files(input_dir)
        total_files = len(files)

        if total_files == 0:
            self.logger.warning("No files found to process!")
            return self._create_empty_metrics()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Process files
        self.logger.info(f"\nProcessing {total_files} files...")
        self.logger.info("-"*80)

        max_concurrent = self.config['general']['max_concurrent_processes']
        error_handling = self.config.get('error_handling', {})
        skip_failed = error_handling.get('skip_failed_files', True)
        max_retries = error_handling.get('max_retries', 2)

        processed_count = 0
        failed_count = 0

        # Process files with progress tracking
        for i, filepath in enumerate(files, 1):
            # Progress indicator
            progress_percent = (i / total_files) * 100
            elapsed = time.time() - batch_start
            avg_time = elapsed / i if i > 0 else 0
            eta_seconds = avg_time * (total_files - i)
            eta_minutes = eta_seconds / 60

            self.logger.info(
                f"\n[{i}/{total_files}] ({progress_percent:.1f}%) "
                f"ETA: {eta_minutes:.1f} min"
            )

            # Process with retries
            result = None
            for attempt in range(max_retries + 1):
                result = self.process_single_file(
                    filepath,
                    output_dir,
                    input_dir,
                    preset
                )

                if result.success:
                    break
                elif attempt < max_retries:
                    retry_delay = error_handling.get('retry_delay_seconds', 5)
                    self.logger.warning(
                        f"Retrying {result.filename} (attempt {attempt + 2}/{max_retries + 1}) "
                        f"in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)

            # Track results
            with self.processing_lock:
                self.results.append(result)

                if result.success:
                    processed_count += 1
                    self.stats['processed'] += 1
                    self.stats['total_chars'] += result.char_count
                    self.stats['total_words'] += result.word_count
                else:
                    failed_count += 1
                    self.stats['failed'] += 1
                    self.failed_files.append((filepath, result.error_message))

                    if not skip_failed:
                        self.logger.error("Stopping batch due to error (skip_failed_files=false)")
                        break

                # Track GPU memory
                self.stats['gpu_memory_samples'].append(self.gpu_monitor.current_usage_mb)

            # Clear GPU cache periodically
            if self.config['gpu'].get('clear_cache_between_batches', True):
                if i % 10 == 0:  # Every 10 files
                    self.gpu_monitor.clear_cache()

        # Generate batch metrics
        metrics = self._create_metrics(batch_start, total_files)

        # Save error log
        self._save_error_log()

        # Save metrics summary
        self._save_metrics_summary(metrics)

        # Print summary
        self._print_summary(metrics)

        return metrics

    def _create_metrics(self, start_time: float, total_files: int) -> BatchMetrics:
        """Create batch metrics from collected statistics"""
        total_time = time.time() - start_time

        successful = self.stats['processed']
        failed = self.stats['failed']
        skipped = self.stats['skipped']

        avg_time = total_time / max(successful, 1)
        throughput = (successful / total_time) * 60 if total_time > 0 else 0

        gpu_samples = self.stats['gpu_memory_samples']
        peak_gpu = max(gpu_samples) if gpu_samples else 0
        avg_gpu = sum(gpu_samples) / len(gpu_samples) if gpu_samples else 0

        error_rate = (failed / max(total_files, 1)) * 100

        return BatchMetrics(
            total_files=total_files,
            successful=successful,
            failed=failed,
            skipped=skipped,
            total_time_seconds=total_time,
            average_time_per_file=avg_time,
            throughput_files_per_minute=throughput,
            peak_gpu_memory_mb=peak_gpu,
            average_gpu_memory_mb=avg_gpu,
            total_chars_extracted=self.stats['total_chars'],
            total_words_extracted=self.stats['total_words'],
            error_rate_percent=error_rate,
            start_time=self.stats['start_time'],
            end_time=datetime.utcnow().isoformat(),
            failed_files=[f[0] for f in self.failed_files]
        )

    def _create_empty_metrics(self) -> BatchMetrics:
        """Create empty metrics when no files are processed"""
        return BatchMetrics(
            total_files=0,
            successful=0,
            failed=0,
            skipped=0,
            total_time_seconds=0,
            average_time_per_file=0,
            throughput_files_per_minute=0,
            peak_gpu_memory_mb=0,
            average_gpu_memory_mb=0,
            total_chars_extracted=0,
            total_words_extracted=0,
            error_rate_percent=0,
            start_time=datetime.utcnow().isoformat(),
            end_time=datetime.utcnow().isoformat(),
            failed_files=[]
        )

    def _save_error_log(self):
        """Save error log to file"""
        if not self.failed_files:
            return

        error_config = self.config.get('error_handling', {})
        error_log_file = error_config.get('error_log_file', 'batch_errors.log')

        with open(error_log_file, 'w', encoding='utf-8') as f:
            f.write("DeepSeek-OCR Batch Processing Errors\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {datetime.utcnow().isoformat()}\n")
            f.write(f"Total errors: {len(self.failed_files)}\n\n")

            for filepath, error in self.failed_files:
                f.write(f"File: {filepath}\n")
                f.write(f"Error: {error}\n")
                f.write("-"*80 + "\n")

        self.logger.info(f"Error log saved to: {error_log_file}")

        # Also save failed files list as CSV
        failed_list_file = error_config.get('failed_files_list', 'failed_files.csv')
        with open(failed_list_file, 'w', encoding='utf-8') as f:
            f.write("filepath,error\n")
            for filepath, error in self.failed_files:
                # Escape commas in error message
                error_escaped = error.replace('"', '""') if error else ""
                f.write(f'"{filepath}","{error_escaped}"\n')

        self.logger.info(f"Failed files list saved to: {failed_list_file}")

    def _save_metrics_summary(self, metrics: BatchMetrics):
        """Save metrics summary to JSON file"""
        log_config = self.config.get('logging', {})
        metrics_file = log_config.get('metrics_summary_file', 'batch_metrics.json')

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)

        self.logger.info(f"Metrics summary saved to: {metrics_file}")

    def _print_summary(self, metrics: BatchMetrics):
        """Print batch processing summary"""
        self.logger.info("\n" + "="*80)
        self.logger.info("BATCH PROCESSING COMPLETED")
        self.logger.info("="*80)
        self.logger.info(f"Total files:     {metrics.total_files}")
        self.logger.info(f"Successful:      {metrics.successful} ({(metrics.successful/max(metrics.total_files,1)*100):.1f}%)")
        self.logger.info(f"Failed:          {metrics.failed}")
        self.logger.info(f"Skipped:         {metrics.skipped}")
        self.logger.info(f"Total time:      {metrics.total_time_seconds/60:.1f} minutes")
        self.logger.info(f"Avg time/file:   {metrics.average_time_per_file:.1f} seconds")
        self.logger.info(f"Throughput:      {metrics.throughput_files_per_minute:.1f} files/minute")
        self.logger.info(f"Peak GPU memory: {metrics.peak_gpu_memory_mb:.0f} MB")
        self.logger.info(f"Avg GPU memory:  {metrics.average_gpu_memory_mb:.0f} MB")
        self.logger.info(f"Total chars:     {metrics.total_chars_extracted:,}")
        self.logger.info(f"Total words:     {metrics.total_words_extracted:,}")
        self.logger.info(f"Error rate:      {metrics.error_rate_percent:.1f}%")
        self.logger.info("="*80)

        # Performance assessment
        target_throughput = self.config['performance'].get('target_throughput', 75)
        if metrics.throughput_files_per_minute >= target_throughput:
            self.logger.info(f"✓ Performance target met ({target_throughput} files/min)")
        else:
            self.logger.warning(
                f"⚠ Performance below target: "
                f"{metrics.throughput_files_per_minute:.1f} vs {target_throughput} files/min"
            )
