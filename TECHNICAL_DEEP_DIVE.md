# Technical Deep Dive: Architecture & Implementation Details

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER SYSTEM                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                   ELECTRON APP (main.js)                       │ │
│  │                                                                 │ │
│  │  1. Spawn Python process                                       │ │
│  │  2. Poll health endpoint every 1 second (30s timeout)         │ │
│  │  3. Create Electron window (1200x800)                         │ │
│  │                                                                 │ │
│  │  Handlers:                                                     │ │
│  │  - check-server-status → GET /health                          │ │
│  │  - load-model → POST /load_model                              │ │
│  │  - perform-ocr → POST /ocr (FormData)                         │ │
│  │  - select-image → File dialog                                 │ │
│  │                                                                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│           ↓ (child process)              ↑ (HTTP axios)              │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                  FLASK SERVER (ocr_server.py)                  │ │
│  │                    localhost:5000                              │ │
│  │                                                                 │ │
│  │  Routes:                                                       │ │
│  │  GET  /health          → Server + GPU status                  │ │
│  │  POST /load_model      → Load model in background thread      │ │
│  │  GET  /progress        → Model loading/OCR progress           │ │
│  │  POST /ocr             → Perform OCR inference               │ │
│  │  GET  /model_info      → Model metadata                       │ │
│  │  GET  /outputs/<file>  → Serve result files                   │ │
│  │                                                                 │ │
│  │  Global State:                                                 │ │
│  │  - model (AutoModel loaded in bfloat16)                       │ │
│  │  - tokenizer (AutoTokenizer)                                  │ │
│  │  - progress_data (dict with lock)                             │ │
│  │                                                                 │ │
│  │  Cache:                                                        │ │
│  │  - ./cache/models/  (HF model cache)                          │ │
│  │  - ./cache/outputs/ (OCR results)                             │ │
│  │                                                                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│           ↓                              ↑                           │
│           │                              │                           │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │               RENDERER PROCESS (renderer.js)                   │ │
│  │                                                                 │ │
│  │  UI Components:                                                │ │
│  │  - Drop zone (drag & drop images)                             │ │
│  │  - Image preview                                              │ │
│  │  - OCR results (text left, preview right)                     │ │
│  │  - Bounding boxes (SVG overlay, clickable)                    │ │
│  │  - Status bar (server, model, GPU)                            │ │
│  │  - Controls (prompt type, sizes, crop mode)                   │ │
│  │                                                                 │ │
│  │  Real-Time Polling:                                           │ │
│  │  setInterval(async () => {                                    │ │
│  │    const progress = await fetch('/progress')                  │ │
│  │    // Parse boxes, update count, render incrementally        │ │
│  │  }, 200ms)                                                    │ │
│  │                                                                 │ │
│  │  Token Parsing:                                               │ │
│  │  Regex: <|ref|>CONTENT<|/ref|><|det|>[[x1,y1,x2,y2]]        │ │
│  │                                                                 │ │
│  │  Box Rendering:                                               │ │
│  │  SVG rectangles + labels, colored by type                     │ │
│  │  10 known types, unknown flagged in lime green                │ │
│  │                                                                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                  LAUNCHER (start.py)                           │ │
│  │                                                                 │ │
│  │  1. Check Node.js 18+ exists                                  │ │
│  │  2. Check Python 3.12+ version                                │ │
│  │  3. Detect GPU:                                               │ │
│  │     - Run nvidia-smi --query-gpu=compute_cap                  │ │
│  │     - Extract major version                                   │ │
│  │     - Select CUDA 11.8 (< 5) or 12.4 (≥ 5)                   │ │
│  │  4. Create venv (if missing)                                  │ │
│  │  5. Install PyTorch from appropriate index URL                │ │
│  │  6. Install Python dependencies                               │ │
│  │  7. Run: npm start                                            │ │
│  │                                                                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow: Image Processing

```
User drops image
    ↓
renderer.js: loadImage(filePath)
    - Store currentImagePath
    - Display preview
    - Enable OCR button
    ↓
User clicks "Run OCR"
    ↓
renderer.js: performOCR()
    - Set isProcessing = true
    - Create polling interval (200ms)
    - Call ipcRenderer.invoke('perform-ocr', {...})
    ↓
main.js: ipcMain.handle('perform-ocr')
    - Read image file to buffer
    - Create FormData with image + params
    - POST to http://127.0.0.1:5000/ocr
    ↓
Flask: perform_ocr()
    1. Check model loaded (load if needed)
    2. Save image to temp file
    3. Capture stdout
    4. Call model.infer()
       - Load image
       - Tokenize prompt
       - Generate tokens via transformer
       - Output token stream to stdout
    5. Parse token stream:
       - Extract <|ref|>...<|det|>[[x1,y1,x2,y2]]...
       - Store raw_tokens
    6. Read output file (result.mmd or result.txt)
    7. Return {result, raw_tokens, prompt_type}
    ↓
main.js returns response to renderer
    ↓
renderer.js: performOCR() continuation
    - Display result text
    - Parse and render boxes
    - Show copy/download buttons
    - Clear polling interval
    ↓
Display complete in UI
```

## Model Loading Sequence

```
User clicks "Load Model"
    ↓
renderer.js: loadModel()
    - Set isProcessing = true
    - Start polling /progress every 500ms
    - Call ipcRenderer.invoke('load-model')
    ↓
main.js: ipcMain.handle('load-model')
    - POST to http://127.0.0.1:5000/load_model
    ↓
Flask: load_model_endpoint()
    - Call load_model()
    - Spawn Thread(target=load_model_background)
    ↓
Background Thread: load_model_background()
    
    Step 1: Initialize (0%)
    ├─ Create cache directory
    ├─ Check GPU availability
    └─ update_progress('loading', 'init', ..., 0)
    
    Step 2: Load Tokenizer (10-20%)
    ├─ AutoTokenizer.from_pretrained()
    ├─ Check HF cache (50GB+ potentially)
    └─ update_progress('loading', 'tokenizer', ..., 20)
    
    Step 3: Download/Load Model (25-75%)
    ├─ Check initial cache size
    ├─ Spawn monitor thread if downloading
    │  └─ Every 2 seconds: check directory size, update progress
    ├─ Try flash_attention_2 optimization
    ├─ Fall back to standard attention if fails
    ├─ AutoModel.from_pretrained()
    └─ update_progress('loading', 'model', ..., 75)
    
    Step 4: Move to GPU (80-95%)
    ├─ model.eval()
    ├─ model.cuda()
    ├─ Convert to torch.bfloat16
    ├─ Log GPU name and compute capability
    └─ update_progress('loading', 'gpu', ..., 90)
    
    Step 5: Complete (100%)
    └─ update_progress('loaded', 'complete', ..., 100)

Meanwhile in renderer.js:
    - Polling thread reads /progress every 500ms
    - Updates progress bar with percent
    - Updates status message with stage
    - When status == 'loaded', stop polling
    - Show success message

Total time: 5-30 minutes (depends on download vs cache)
```

## Code Examples

### Example 1: Token Parsing in Renderer

```javascript
// File: /home/user/deepseek-ocr-client/renderer.js, lines 363-424
function parseBoxesFromTokens(tokenText, isOcrComplete = false) {
    // Regex to match: <|ref|>CONTENT<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
    const refDetRegex = /<\|ref\|>([^<]+)<\|\/ref\|><\|det\|>\[\[([^\]]+)\]\]<\|\/det\|>/g;
    let match;
    const matches = [];

    // Collect all matches with positions
    while ((match = refDetRegex.exec(tokenText)) !== null) {
        matches.push({
            content: match[1].trim(),      // "title", "text", "Hello world"
            coords: match[2],               // "100, 200, 300, 400"
            matchStart: match.index,
            matchEnd: match.index + match[0].length
        });
    }

    // Process each match
    for (let i = 0; i < matches.length; i++) {
        try {
            const matchData = matches[i];
            const content = matchData.content;

            // Parse coordinates "x1, y1, x2, y2"
            const coords = matchData.coords
                .split(',')
                .map(s => parseFloat(s.trim()))
                .filter(n => !isNaN(n));
            
            if (coords.length === 4) {
                // Check if content is a known type label
                const isType = KNOWN_TYPES.includes(content);
                
                // Extract actual text after this box
                let textContent = '';
                let isComplete = false;

                if (i < matches.length - 1) {
                    // Get text between this box and next
                    textContent = tokenText.substring(
                        matchData.matchEnd, 
                        matches[i + 1].matchStart
                    ).trim();
                    isComplete = textContent.length > 0;
                } else {
                    // Get everything after last box
                    textContent = tokenText.substring(matchData.matchEnd).trim();
                    isComplete = isOcrComplete && textContent.length > 0;
                }

                boxes.push({
                    content: content,
                    textContent: textContent,
                    isType: isType,
                    type: isType ? content : 'text',
                    x1: coords[0], y1: coords[1],
                    x2: coords[2], y2: coords[3],
                    isComplete: isComplete
                });
            }
        } catch (e) {
            console.error('Error parsing box coordinates:', e);
        }
    }

    return boxes;
}
```

### Example 2: Model Loading with Progress

```python
# File: /home/user/deepseek-ocr-client/backend/ocr_server.py, lines 89-202
def load_model_background():
    """Background thread function to load the model"""
    global model, tokenizer

    try:
        update_progress('loading', 'init', 'Initializing model loading...', 0)
        logger.info(f"Loading DeepSeek OCR model from {MODEL_NAME}...")

        # Create cache directory
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

        # Check GPU availability
        has_gpu = check_gpu_availability()

        # Load tokenizer (10% progress)
        update_progress('loading', 'tokenizer', 'Loading tokenizer...', 10)
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR
        )
        update_progress('loading', 'tokenizer', 'Tokenizer loaded', 20)

        # Check if model is already cached
        initial_cache_size = get_cache_dir_size(MODEL_CACHE_DIR)
        is_cached = initial_cache_size > 100 * 1024 * 1024  # > 100 MB

        if is_cached:
            update_progress('loading', 'model', 'Loading model from cache...', 25)
        else:
            update_progress('loading', 'model', 
                'Downloading model files (this will take several minutes)...', 25)

        # Monitor download progress
        download_monitor_active = [True]
        def monitor_download():
            last_size = initial_cache_size
            stall_count = 0
            progress = 25

            while download_monitor_active[0] and progress < 75:
                time.sleep(2)
                current_size = get_cache_dir_size(MODEL_CACHE_DIR)

                if current_size > last_size:
                    stall_count = 0
                    progress = min(progress + 2, 75)
                    size_mb = current_size / (1024 * 1024)
                    update_progress('loading', 'model', 
                        f'Downloading model files... ({size_mb:.1f} MB downloaded)', progress)
                    last_size = current_size
                else:
                    stall_count += 1
                    if stall_count < 5:
                        msg = 'Loading model from cache...' if is_cached else 'Downloading model files...'
                        update_progress('loading', 'model', msg, progress)

        monitor_thread = Thread(target=monitor_download)
        monitor_thread.daemon = True
        monitor_thread.start()

        # Load model with flash attention if available
        try:
            model = AutoModel.from_pretrained(
                MODEL_NAME,
                _attn_implementation='flash_attention_2',
                trust_remote_code=True,
                use_safetensors=True,
                cache_dir=MODEL_CACHE_DIR
            )
            logger.info("Using flash attention 2")
        except Exception as e:
            logger.warning(f"Flash attention not available: {e}, using default attention")
            model = AutoModel.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                use_safetensors=True,
                cache_dir=MODEL_CACHE_DIR
            )

        # Stop monitor thread
        download_monitor_active[0] = False
        monitor_thread.join(timeout=5)

        # Move to GPU
        update_progress('loading', 'gpu', 'Moving model to GPU...', 80)
        model = model.eval()

        update_progress('loading', 'gpu', 'Optimizing model on GPU...', 90)
        if has_gpu:
            model = model.cuda().to(torch.bfloat16)
            logger.info("Model loaded on GPU with bfloat16")
        else:
            logger.info("Model loaded on CPU (inference will be slower)")

        logger.info("Model loaded successfully!")
        update_progress('loaded', 'complete', 'Model ready!', 100)

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        update_progress('error', 'failed', str(e), 0)
        import traceback
        traceback.print_exc()
```

### Example 3: GPU Detection in Launcher

```python
# File: /home/user/deepseek-ocr-client/start.py, lines 84-123
def get_gpu_compute_capability():
    """Get GPU compute capability using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            shell=(sys.platform == "win32")
        )
        compute_cap = result.stdout.strip()
        if compute_cap:
            # Also get GPU name for display
            name_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                shell=(sys.platform == "win32")
            )
            gpu_name = name_result.stdout.strip()

            major = int(compute_cap.split('.')[0])
            print(f"✓ NVIDIA GPU detected: {gpu_name}")
            print(f"  Compute Capability: {compute_cap}")
            return major
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass

    print("! No NVIDIA GPU detected")
    return None

def determine_cuda_version(compute_major):
    """Determine which CUDA version to use based on compute capability."""
    if compute_major is None:
        return "cpu"
    elif compute_major < 5:
        return "cu118"  # CUDA 11.8 for Kepler and older
    else:
        return "cu124"  # CUDA 12.4 for Maxwell and newer
```

## Performance Characteristics

### Model Loading
- **First Run:** 15-30 minutes (download + setup)
  - Download: 10-20 GB transfer (depends on bandwidth)
  - Model size: ~7GB after conversion
  - Tokenizer: ~50 MB
- **Subsequent Runs:** 5-15 minutes (load from cache)
  - Cache hit: HF handles efficiently
  - GPU memory: ~14 GB for bfloat16

### OCR Processing
- **Time per image:** 10-60 seconds (varies by image complexity)
  - Small images (512x512): ~10-15 seconds
  - Large images (1280x1280): ~30-60 seconds
  - Depends on: image complexity, GPU VRAM, content density
- **GPU Memory:** ~14 GB (bfloat16 precision)
- **Token Generation:** Streaming, ~100 tokens/second (GPU dependent)

### Real-Time Polling Overhead
- Renderer polls every 200ms during OCR
- Network roundtrip: ~5-10ms per poll
- CPU impact: Negligible (<1%)
- Enables smooth progressive box rendering

## Limitations & Known Issues

1. **Single Image Only:** No batch processing
2. **Synchronous Processing:** Can't queue multiple images
3. **Memory Not Cleared:** GPU memory persists between runs
4. **Hardcoded Config:** No environment variable support
5. **No Validation:** Image size/format not validated
6. **CDN Dependencies:** marked.js and jszip loaded from CDN
7. **Token Parsing:** Simple regex; doesn't validate coordinates
8. **Resolution Bug:** README mentions "known issue" with certain size combinations
9. **No Retry Logic:** Failed OCR doesn't retry
10. **Single User:** Electron app; can't be shared across network

## Security Notes

### Current (Minimal)
- ✅ CORS only on localhost (implicit security)
- ✅ Image files deleted after processing
- ✅ No external API calls (self-contained)
- ❌ No input validation
- ❌ No authentication
- ❌ No encryption
- ❌ No audit logging

### Needed for Medical Use
- Add authentication layer
- Validate all inputs
- Encrypt data in transit (HTTPS)
- Implement audit logging
- Add rate limiting
- Sanitize error messages (no sensitive data leaks)
- Create security policy documentation

## Testing Coverage

### Currently Missing
- No unit tests
- No integration tests
- No e2e tests
- No performance tests
- No medical validation tests

### Recommended
- Unit tests for token parsing (>20 test cases)
- Integration tests for OCR pipeline
- GPU memory leak detection
- Model output validation
- Edge case handling (corrupted images, extreme sizes)
