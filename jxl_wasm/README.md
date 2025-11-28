# JXL WASM Decoder

A WebAssembly decoder for JPEG XL images that runs entirely in the browser. This allows you to use JXL images on the web without relying on browser support.

## Features

- ✅ Decode JXL images to PNG/APNG in the browser
- ✅ Support for animated JXL images (outputs APNG)
- ✅ Preserve color profiles (ICC, sRGB, etc.)
- ✅ Zero dependencies on browser JXL support
- ✅ Fast Rust-based decoder compiled to WASM
- ✅ Small bundle size with optimized builds

## Building

### Prerequisites

1. Install Rust: https://rustup.rs/
2. Install wasm-pack:
   ```bash
   cargo install wasm-pack
   ```

### Build the WASM module

```bash
# From the jxl_wasm directory
wasm-pack build --target web --release

# Or for development (with debug symbols)
wasm-pack build --target web --dev
```

This will create a `pkg` directory containing:
- `jxl_wasm_bg.wasm` - The compiled WASM binary
- `jxl_wasm.js` - JavaScript bindings
- `jxl_wasm.d.ts` - TypeScript definitions

### Optimized build

For production, you can further optimize the WASM binary:

```bash
wasm-pack build --target web --release

# Optional: Optimize with wasm-opt (from binaryen)
wasm-opt -Oz -o pkg/jxl_wasm_bg_optimized.wasm pkg/jxl_wasm_bg.wasm
```

## Usage

### Basic Example

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>JXL Decoder</title>
</head>
<body>
    <input type="file" id="file-input" accept=".jxl">
    <img id="output">

    <script type="module">
        import init, { decode_jxl_to_png } from './pkg/jxl_wasm.js';

        async function main() {
            // Initialize the WASM module
            await init();

            document.getElementById('file-input').addEventListener('change', async (e) => {
                const file = e.target.files[0];
                if (!file) return;

                // Read the JXL file
                const arrayBuffer = await file.arrayBuffer();
                const jxlData = new Uint8Array(arrayBuffer);

                // Decode to PNG
                const pngData = decode_jxl_to_png(jxlData);

                // Display the image
                const blob = new Blob([pngData], { type: 'image/png' });
                const url = URL.createObjectURL(blob);
                document.getElementById('output').src = url;
            });
        }

        main();
    </script>
</body>
</html>
```

### TypeScript Example

```typescript
import init, { decode_jxl_to_png, get_jxl_info, init_panic_hook } from './pkg/jxl_wasm';

async function decodeJXL(jxlData: Uint8Array): Promise<Blob> {
    // Initialize WASM (call once at startup)
    await init();
    init_panic_hook(); // Better error messages in console

    // Decode JXL to PNG bytes
    const pngData = decode_jxl_to_png(jxlData);

    // Create a blob that can be used as img src
    return new Blob([pngData], { type: 'image/png' });
}

async function getImageInfo(jxlData: Uint8Array) {
    await init();

    const info = get_jxl_info(jxlData);
    console.log(`Dimensions: ${info.width} x ${info.height}`);
    console.log(`Frames: ${info.num_frames}`);
    console.log(`Has alpha: ${info.has_alpha}`);
}
```

### React Example

```tsx
import { useEffect, useState } from 'react';
import init, { decode_jxl_to_png } from './pkg/jxl_wasm';

function JXLImage({ jxlUrl }: { jxlUrl: string }) {
    const [imageSrc, setImageSrc] = useState<string>('');

    useEffect(() => {
        let mounted = true;

        async function loadImage() {
            // Initialize WASM
            await init();

            // Fetch JXL file
            const response = await fetch(jxlUrl);
            const arrayBuffer = await response.arrayBuffer();
            const jxlData = new Uint8Array(arrayBuffer);

            // Decode to PNG
            const pngData = decode_jxl_to_png(jxlData);
            const blob = new Blob([pngData], { type: 'image/png' });
            const url = URL.createObjectURL(blob);

            if (mounted) {
                setImageSrc(url);
            }
        }

        loadImage().catch(console.error);

        return () => {
            mounted = false;
            if (imageSrc) {
                URL.revokeObjectURL(imageSrc);
            }
        };
    }, [jxlUrl]);

    return imageSrc ? <img src={imageSrc} alt="JXL image" /> : <div>Loading...</div>;
}
```

## API Reference

### `decode_jxl_to_png(jxl_data: Uint8Array): Uint8Array`

Decodes a JXL image and returns PNG bytes.

**Parameters:**
- `jxl_data`: The JXL image data as a Uint8Array

**Returns:**
- PNG image data as a Uint8Array (APNG for animated images)

**Throws:**
- JsValue with error message if decoding fails

### `get_jxl_info(jxl_data: Uint8Array): ImageInfo`

Gets basic information about a JXL image without fully decoding it.

**Parameters:**
- `jxl_data`: The JXL image data as a Uint8Array

**Returns:**
- `ImageInfo` object with:
  - `width: number` - Image width in pixels
  - `height: number` - Image height in pixels
  - `num_frames: number` - Number of frames (1 for still images)
  - `has_alpha: boolean` - Whether the image has an alpha channel

### `init_panic_hook(): void`

Sets up better error messages in the browser console. Call once at startup.

## Demo

A complete demo is included in the `demo` directory. To run it:

1. Build the WASM module (see above)
2. Copy the `pkg` directory into the `demo` directory:
   ```bash
   cp -r pkg demo/
   ```
3. Serve the demo directory with a web server:
   ```bash
   # Using Python 3
   cd demo
   python3 -m http.server 8000

   # Or using Node.js
   npx http-server demo -p 8000
   ```
4. Open http://localhost:8000 in your browser

## Performance

The decoder is built on the high-performance `jxl-rs` library and includes optimizations from the "road to 1x or less" PR:

- SIMD optimizations (compiled out for WASM compatibility)
- Efficient memory management
- Zero-copy where possible

Typical decode times on a modern machine:
- Small images (< 1MP): 10-50ms
- Medium images (1-5MP): 50-200ms
- Large images (> 5MP): 200-1000ms

## Bundle Size

Optimized WASM binary size: ~500-800KB (gzipped: ~200-300KB)

## Browser Compatibility

Works in all modern browsers that support WebAssembly:
- Chrome 57+
- Firefox 52+
- Safari 11+
- Edge 79+

## Limitations

- No SIMD acceleration in WASM (yet - waiting for WASM SIMD to be more widely supported)
- No parallel decoding (single-threaded in browser)
- Memory constraints of browser environment

## License

BSD-3-Clause (same as jxl-rs)
