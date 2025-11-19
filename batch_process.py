#!/usr/bin/env python3
"""
DeepSeek-OCR Batch Processing CLI
Command-line interface for batch processing medical documents

Usage:
    python batch_process.py --input ./docs --output ./processed
    python batch_process.py --input ./docs --output ./processed --preset laboratorio
    python batch_process.py --input ./docs --output ./processed --config custom_config.yaml

Examples for Medical Documents (Peru):
    # Process historias clínicas
    python batch_process.py --input ./historias_clinicas --output ./processed_hc --preset historia_clinica

    # Process lab results
    python batch_process.py --input ./laboratorios --output ./processed_labs --preset laboratorio

    # Process prescriptions
    python batch_process.py --input ./recetas --output ./processed_rx --preset receta

    # Process mixed medical documents (auto-detect)
    python batch_process.py --input ./documentos_medicos --output ./processed --preset generic
"""

import os
import sys
import argparse
from pathlib import Path

# Add backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from batch_processor import BatchProcessor


def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║          DeepSeek-OCR Batch Processor v1.0                      ║
║          Medical Document Processing System                      ║
║                                                                  ║
║          Optimized for RTX 5090 GPU (32GB VRAM)                 ║
║          Target: 50-100 documents/minute                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_presets_info(config_path: str = "batch_config.yaml"):
    """Print available medical presets"""
    import yaml

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        presets = config.get('medical_presets', {})

        print("\n📋 Available Medical Presets:\n")
        print("-" * 70)

        for name, preset in presets.items():
            desc = preset.get('description', 'No description')
            prompt_type = preset.get('prompt_type', 'N/A')
            print(f"  {name:20s} - {desc}")
            print(f"  {'':20s}   (Mode: {prompt_type})")
            print()

        print("-" * 70)
        print()

    except Exception as e:
        print(f"Warning: Could not load presets: {e}")


def validate_paths(args):
    """Validate input/output paths"""
    # Check input directory exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input directory does not exist: {args.input}")
        sys.exit(1)

    if not os.path.isdir(args.input):
        print(f"❌ Error: Input path is not a directory: {args.input}")
        sys.exit(1)

    # Check if input directory is empty
    if not any(os.scandir(args.input)):
        print(f"⚠️  Warning: Input directory is empty: {args.input}")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(0)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        print(f"📁 Creating output directory: {args.output}")
        os.makedirs(args.output, exist_ok=True)

    # Check if output directory is not empty
    if os.path.exists(args.output) and any(os.scandir(args.output)):
        print(f"⚠️  Warning: Output directory is not empty: {args.output}")
        if not args.force:
            response = input("Files may be overwritten. Continue? (y/N): ")
            if response.lower() != 'y':
                sys.exit(0)

    # Check config file exists
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file does not exist: {args.config}")
        sys.exit(1)


def print_processing_info(args):
    """Print processing information"""
    print(f"\n🚀 Starting Batch Processing\n")
    print(f"  Input directory:  {os.path.abspath(args.input)}")
    print(f"  Output directory: {os.path.abspath(args.output)}")
    print(f"  Config file:      {args.config}")
    print(f"  Medical preset:   {args.preset}")
    print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR Batch Processor for Medical Documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process folder of medical documents with default preset
  %(prog)s --input ./docs --output ./processed

  # Process historias clínicas
  %(prog)s --input ./hc --output ./processed --preset historia_clinica

  # Process lab results with custom config
  %(prog)s --input ./labs --output ./processed --preset laboratorio --config my_config.yaml

  # Force overwrite existing output
  %(prog)s --input ./docs --output ./processed --force

Medical Presets:
  historia_clinica  - Full medical history documents
  laboratorio       - Laboratory results with tables
  receta            - Medical prescriptions (handwritten)
  radiologia        - Radiology reports
  generic           - Generic medical documents (default)

For more information, see README_BATCH.md
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input directory containing medical documents (images/PDFs)',
        metavar='DIR'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for processed results',
        metavar='DIR'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='batch_config.yaml',
        help='Path to configuration file (default: batch_config.yaml)',
        metavar='FILE'
    )

    parser.add_argument(
        '--preset', '-p',
        type=str,
        default='generic',
        choices=['historia_clinica', 'laboratorio', 'receta', 'radiologia', 'generic'],
        help='Medical document preset (default: generic)',
        metavar='PRESET'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force overwrite existing output directory'
    )

    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List available medical presets and exit'
    )

    parser.add_argument(
        '--version', '-v',
        action='version',
        version='DeepSeek-OCR Batch Processor v1.0'
    )

    # Parse arguments
    args = parser.parse_args()

    # Print banner
    print_banner()

    # Handle --list-presets
    if args.list_presets:
        print_presets_info(args.config)
        sys.exit(0)

    # Validate paths
    validate_paths(args)

    # Print processing info
    print_processing_info(args)

    # Confirm before processing
    if not args.force:
        response = input("Ready to start processing? (Y/n): ")
        if response.lower() == 'n':
            print("Cancelled.")
            sys.exit(0)

    print()

    try:
        # Initialize batch processor
        print("🔧 Initializing batch processor...")
        processor = BatchProcessor(config_path=args.config)

        # Process batch
        print("⚡ Processing documents...\n")
        metrics = processor.process_batch(
            input_dir=args.input,
            output_dir=args.output,
            preset=args.preset
        )

        # Success
        if metrics.failed == 0:
            print("\n✅ Batch processing completed successfully!")
        else:
            print(f"\n⚠️  Batch processing completed with {metrics.failed} errors")
            print(f"   See batch_errors.log and failed_files.csv for details")

        # Print summary stats
        print(f"\n📊 Summary:")
        print(f"   Processed:  {metrics.successful}/{metrics.total_files} files")
        print(f"   Throughput: {metrics.throughput_files_per_minute:.1f} files/minute")
        print(f"   Total time: {metrics.total_time_seconds/60:.1f} minutes")
        print(f"   Extracted:  {metrics.total_chars_extracted:,} characters")

        # Exit code based on results
        if metrics.failed > 0:
            sys.exit(1)  # Partial failure
        else:
            sys.exit(0)  # Success

    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
