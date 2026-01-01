# Demo

Browser demos for the JXL WASM decoder.

## Run

```bash
python3 -m http.server 8000
```

- **Main demo**: http://localhost:8000 - Drag & drop JXL files
- **Polyfill demo**: http://localhost:8000/polyfill-demo.html - Automatic JXL support

## Requirements

WASM module must be built first:
```bash
cd ..
./build.sh
```

This copies `pkg/` to `demo/pkg/`.

## Polyfill Usage

Automatically handle JXL images on any page:

```html
<script type="module" src="polyfill.js"></script>

<!-- JXL images work automatically -->
<img src="image.jxl" alt="My image">
```

### Options

```javascript
import { JXLPolyfill } from './polyfill.js';

const polyfill = new JXLPolyfill({
  patchImageConstructor: true,  // Intercept new Image() (default: true)
  showLoadingState: true,        // Blur while decoding (default: true)
  cacheDecoded: true,            // Cache PNG blobs (default: true)
  verbose: false                 // Debug logging (default: false)
});

polyfill.start();
```

Add `?jxl-debug` to URL for verbose logging.

### Features

- Detects native JXL support (skips polyfill if browser supports JXL)
- Scans existing `<img src="*.jxl">` tags
- Watches for dynamically added images (MutationObserver)
- Patches `new Image()` constructor
- Caches decoded images
- Shows loading state during decode

## Test Images

- https://github.com/libjxl/conformance
- https://jpegxl.info/images/dice.jxl
- https://jpegxl.info/images/anim-icos.jxl
