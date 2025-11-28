# JXL WASM Decoder Demo

This is a live demo of the JXL WASM decoder running in your browser.

## Running the Demo

1. Start a local web server in this directory:

   ```bash
   # Using Python 3
   python3 -m http.server 8000

   # Or using Node.js
   npx http-server . -p 8000

   # Or using PHP
   php -S localhost:8000
   ```

2. Open http://localhost:8000 in your browser

3. Drag and drop a JXL file or click to browse

## Getting JXL Test Images

You can download sample JXL images from:
- https://github.com/libjxl/conformance
- https://jpegxl.info/
- Convert your own images using cjxl from https://github.com/libjxl/libjxl

## Features Demonstrated

- ✅ Drag and drop file upload
- ✅ JXL to PNG decoding in browser
- ✅ Image information display
- ✅ Performance metrics
- ✅ Download decoded PNG
- ✅ Support for animated JXL (outputs APNG)

## Browser Requirements

Any modern browser with WebAssembly support:
- Chrome 57+
- Firefox 52+
- Safari 11+
- Edge 79+

## Technical Details

- **WASM Size**: ~1.5MB (gzipped: ~500KB)
- **Decoder**: Built on jxl-rs
- **Output Format**: PNG/APNG
- **Color Profiles**: ICC, sRGB, and other color spaces preserved

## Troubleshooting

### CORS Errors

If you see CORS errors, make sure you're serving the files through a web server (not opening index.html directly with file://).

### Module Loading Errors

Ensure the `pkg` directory is present and contains:
- jxl_wasm_bg.wasm
- jxl_wasm.js
- jxl_wasm.d.ts

If these files are missing, rebuild with:
```bash
cd .. && wasm-pack build --target web --release
```
