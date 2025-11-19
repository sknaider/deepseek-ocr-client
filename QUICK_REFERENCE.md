# Quick Reference: DeepSeek-OCR Client

## File Structure (Absolute Paths)

```
/home/user/deepseek-ocr-client/
├── main.js                                 (261 lines) - Electron main process
├── renderer.js                             (1,008 lines) - Electron UI
├── start.py                                (284 lines) - Launcher/installer
├── index.html                              - HTML markup
├── styles.css                              - CSS styling
├── package.json                            - NPM config
├── package-lock.json
├── requirements.txt                        - Python dependencies (16 packages)
├── README.md
├── LICENSE.md
├── start-client.bat                        - Windows launcher
├── start-client.sh                         - Unix launcher
│
├── backend/
│   ├── __init__.py                         - Empty init
│   └── ocr_server.py                       (490 lines) - Flask REST API
│
├── docs/
│   └── images/                             - Documentation images
│
└── .gitignore                              - Ignores: node_modules, cache, venv
```

## Key Line References

### Backend (ocr_server.py)
| Function | Lines | Purpose |
|----------|-------|---------|
| `check_gpu_availability()` | 67-74 | NVIDIA GPU detection |
| `load_model_background()` | 89-202 | Background model loading with progress |
| `perform_ocr()` | 258-461 | Main OCR processing endpoint |
| `CharCountingStream` | 331-380 | Token stream capture |
| `/health` endpoint | 233-240 | Server status check |
| `/progress` endpoint | 242-247 | Progress polling |
| `/ocr` endpoint | 258-461 | Process image |
| `/outputs/<file>` | 474-477 | Serve result files |

### Frontend - Main Process (main.js)
| Function | Lines | Purpose |
|----------|-------|---------|
| `createWindow()` | 12-50 | Create Electron window |
| `startPythonServer()` | 52-140 | Spawn Flask server |
| `checkServerHealth()` | 142-149 | Health check |
| `ipcMain.handle()` | 159-232 | All IPC handlers |
| `check-server-status` handler | 160-167 | - |
| `perform-ocr` handler | 201-232 | Multipart form-data upload |

### Frontend - Renderer (renderer.js)
| Function | Lines | Purpose |
|----------|-------|---------|
| `setupEventListeners()` | 79-138 | UI event wiring |
| `checkServerStatus()` | 140-191 | Poll server status |
| `selectImage()` | 193-199 | File picker |
| `loadImage()` | 201-224 | Display image |
| `parseBoxesFromTokens()` | 363-424 | Parse OCR token stream |
| `renderBoxes()` | 435-598 | Draw SVG boxes |
| `loadModel()` | 600-697 | Load DeepSeek model |
| `performOCR()` | 699-882 | Main OCR processing |
| `displayResults()` | 884-910 | Show results |
| `copyResults()` | 912-925 | Copy to clipboard |
| `downloadZip()` | 927-1000 | Export as ZIP |

### Launcher (start.py)
| Function | Lines | Purpose |
|----------|-------|---------|
| `check_prerequisites()` | 50-69 | Verify Node.js + Python 3.12+ |
| `get_gpu_compute_capability()` | 84-114 | NVIDIA GPU detection |
| `determine_cuda_version()` | 116-123 | Select CUDA 11.8 or 12.4 |
| `setup_python_environment()` | 125-231 | Create venv, install PyTorch |

## Important Constants & Hardcoded Values

```python
# Backend (ocr_server.py)
PYTHON_SERVER_PORT = 5000
PYTHON_SERVER_URL = 'http://127.0.0.1:5000'
MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'
CACHE_DIR = './cache'
MODEL_CACHE_DIR = './cache/models'
OUTPUT_DIR = './cache/outputs'

# Model sizes (from docstring)
base_size: [512, 640, 1024, 1280]
image_size: [512, 640, 1024, 1280]
crop_mode: boolean

# Prompt templates
document: "<image>\n<|grounding|>Convert the document to markdown. "
ocr: "<image>\n<|grounding|>OCR this image. "
free: "<image>\nFree OCR. "
figure: "<image>\nParse the figure. "
describe: "<image>\nDescribe this image in detail. "
```

```javascript
// Frontend (renderer.js)
const DEEPSEEK_COORD_MAX = 999;  // Normalized coordinate space
const KNOWN_TYPES = ['title', 'sub_title', 'text', 'table', 'image', 
                     'image_caption', 'figure', 'caption', 'formula', 'list'];

const TYPE_COLORS = {
    'title': '#8B5CF6',           // Purple
    'sub_title': '#A78BFA',       // Light purple
    'text': '#3B82F6',            // Blue
    'table': '#F59E0B',           // Orange
    'image': '#EC4899',           // Pink
    'figure': '#06B6D4',          // Cyan
    'caption': '#10B981',         // Green
    'image_caption': '#4EC483',   // Light green
    'formula': '#EF4444',         // Red
    'list': '#6366F1'             // Indigo
};

// Polling intervals
Model loading: 500ms
OCR progress: 200ms
Server status check: 5000ms
```

## Flask Endpoints Reference

```
GET /health
  Response: {status: 'ok', model_loaded: bool, gpu_available: bool}
  Error: 500 on failure

POST /load_model
  Request: (no body)
  Response: {status: 'success', message: 'Model loaded successfully'}
  Error: 500 on failure

GET /progress
  Response: {
    status: 'idle|loading|loaded|processing|error',
    stage: string,
    message: string,
    progress_percent: 0-100,
    chars_generated: number,
    raw_token_stream: string,
    timestamp: float
  }

POST /ocr
  Request: multipart/form-data
    - image: file (required)
    - prompt_type: 'document|ocr|free|figure|describe' (default: 'document')
    - base_size: 512|640|1024|1280 (default: 1024)
    - image_size: 512|640|1024|1280 (default: 640)
    - crop_mode: 'true'|'false' (default: 'true')
  
  Response: {
    status: 'success',
    result: string (OCR text),
    boxes_image_path: string or null,
    prompt_type: string,
    raw_tokens: string (token stream)
  }
  
  Error: 400 bad request, 500 server error

GET /model_info
  Response: {
    model_name: string,
    cache_dir: string,
    model_loaded: bool,
    gpu_available: bool,
    gpu_name: string or null
  }

GET /outputs/<filename>
  Returns file from ./cache/outputs/
  Common files: result.mmd, result.txt, result_with_boxes.jpg
  Error: 404 if file not found
```

## IPC Messages (Electron to Flask)

```javascript
// Check server status
const result = await ipcRenderer.invoke('check-server-status');
// Returns: {success: bool, data?: {...}, error?: string}

// Load model
const result = await ipcRenderer.invoke('load-model');
// Returns: {success: bool, data?: {...}, error?: string}

// Get model info
const result = await ipcRenderer.invoke('get-model-info');
// Returns: {success: bool, data?: {...}, error?: string}

// Select image file
const result = await ipcRenderer.invoke('select-image');
// Returns: {success: bool, filePath?: string}

// Perform OCR
const result = await ipcRenderer.invoke('perform-ocr', {
  imagePath: string,
  promptType: string,
  baseSize: number,
  imageSize: number,
  cropMode: boolean
});
// Returns: {success: bool, data?: {...}, error?: string}
```

## Dependencies Matrix

### Node.js/Electron
| Package | Version | Purpose |
|---------|---------|---------|
| electron | ^28.0.0 | Desktop app framework |
| axios | ^1.6.0 | HTTP client |
| marked | (CDN) | Markdown rendering |
| jszip | ^3.10.1 | ZIP file creation |

### Python
| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.6.0 | Deep learning core |
| torchvision | 0.21.0 | Image processing |
| torchaudio | 2.6.0 | Audio (unused) |
| transformers | 4.46.3 | Model loading |
| tokenizers | 0.20.3 | Tokenization |
| flask | >=3.0.0 | REST API |
| flask-cors | >=4.0.0 | CORS support |
| pillow | >=10.0.0 | Image manipulation |
| safetensors | >=0.4.0 | Model format |
| accelerate | >=0.25.0 | Distributed training |
| hf-xet | >=0.1.0 | HF file streaming |
| addict | >=2.4.0 | Config objects |
| easydict | >=1.10 | Dict utilities |
| einops | >=0.7.0 | Tensor ops |
| matplotlib | >=3.7.0 | Plotting (unused) |

## Token Format Specification

**Input Prompt:**
```
<image>
<|grounding|>Convert the document to markdown.
```

**Output Token Stream Format:**
```
<|ref|>TYPE_LABEL<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>ACTUAL_TEXT_CONTENT
<|ref|>TYPE_LABEL<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>ACTUAL_TEXT_CONTENT
...
```

**Example:**
```
<|ref|>title<|/ref|><|det|>[[100, 50, 500, 120]]<|/det|>Welcome to the Document
<|ref|>text<|/ref|><|det|>[[100, 150, 900, 300]]<|/det|>This is paragraph 1 with important information.
<|ref|>table<|/ref|><|det|>[[50, 350, 950, 600]]<|/det|>Table data extracted as markdown
```

**Coordinates:**
- Normalized to 0-999 range
- x1, y1 = top-left corner
- x2, y2 = bottom-right corner
- Map to actual pixels: `pixel = normalized / 999 * image_dimension`

## Performance Targets

| Operation | Target | Actual |
|-----------|--------|--------|
| Model first load | - | 15-30 minutes |
| Model cache load | - | 5-15 minutes |
| OCR small image (512x512) | <30s | 10-15 seconds |
| OCR large image (1280x1280) | <60s | 30-60 seconds |
| Box rendering (10 boxes) | <100ms | ~50-100ms |
| Token polling latency | <500ms | 200-400ms |
| GPU memory (bfloat16) | - | ~14 GB |
| Startup time (post-cache) | <10s | 3-8 seconds |

## Troubleshooting Guide

### Model Loading Hangs
1. Check: `nvidia-smi` command works
2. Check: Internet connectivity for HF download
3. Check: Disk space in `./cache` (needs 50GB+)
4. Check: GPU compute capability (should be ≥ 5 for CUDA 12.4)

### OCR Returns No Results
1. Check: Image file format (jpg, png, gif, bmp, webp supported)
2. Check: Image dimensions (512-1280px recommended)
3. Check: Prompt type selected correctly
4. Check: GPU memory available (kill other GPU processes)

### Bounding Boxes Not Appearing
1. Check: Raw tokens populated in progress
2. Check: Token format matches regex pattern
3. Check: Coordinate values 0-999
4. Check: Browser console for parsing errors

### Server Won't Start
1. Check: Python 3.12+ installed
2. Check: Port 5000 not in use
3. Check: Virtual environment activated
4. Check: Requirements installed: `pip install -r requirements.txt`

## Improvement Priority List

### Critical for Medical Use (Weeks 1-2)
- [ ] Add input validation (image size, format)
- [ ] Implement logging (structured JSON logs)
- [ ] Add error handling/retry logic
- [ ] Create configuration file support (.env)

### High Value (Weeks 3-4)
- [ ] Add PostgreSQL persistence
- [ ] Implement batch processing
- [ ] Add authentication/authorization
- [ ] Create audit logging

### Medium Value (Weeks 5-6)
- [ ] Add medical NER (spaCy)
- [ ] Implement document classification
- [ ] Add FHIR output format
- [ ] Create REST API documentation

### Nice to Have (Weeks 7+)
- [ ] Add web UI (replace Electron)
- [ ] Implement caching layer
- [ ] Add performance monitoring
- [ ] Create admin dashboard

## Git Repository Info

**Current Branch:** `claude/review-ecosystem-assessment-01JdPuUzDjzYBVEFpks1ZXYm`

**Recent Commits:**
```
001e6ed Web version future goal
dae9f1a add disclaimer for resolution bug
0876082 typo
db79bd2 Fix npm not starting 2
5f023af Fix npm not starting
109629b Fix for newer nvidia cards, fixes #1
```

**Repository Metadata:**
- Remote origin available
- Clean working tree (no uncommitted changes at analysis time)
- Multiple branches exist (claude/*, origin/claude)

