# JXL WASM Decoder

WebAssembly decoder for JPEG XL images. Decodes JXL to PNG/APNG in the browser.

## Build

```bash
cargo install wasm-pack
wasm-pack build --target web --release
```

Or use the build script:
```bash
./build.sh
```

Output in `pkg/`:
- `jxl_wasm_bg.wasm` - WASM binary (~1.5MB, ~500KB gzipped)
- `jxl_wasm.js` - JavaScript bindings
- `jxl_wasm.d.ts` - TypeScript definitions

## Usage

### HTML

```html
<script type="module">
  import init, { decode_jxl_to_png } from './pkg/jxl_wasm.js';

  await init();

  const jxlData = new Uint8Array(await file.arrayBuffer());
  const pngData = decode_jxl_to_png(jxlData);
  const blob = new Blob([pngData], { type: 'image/png' });
  img.src = URL.createObjectURL(blob);
</script>
```

### TypeScript

```typescript
import init, { decode_jxl_to_png, get_jxl_info, init_panic_hook } from './pkg/jxl_wasm';

await init();
init_panic_hook(); // optional: better error messages

const pngData = decode_jxl_to_png(jxlData);
const info = get_jxl_info(jxlData);
console.log(`${info.width}x${info.height}, ${info.num_frames} frames`);
```

### React

```tsx
import { useEffect, useState } from 'react';
import init, { decode_jxl_to_png } from './pkg/jxl_wasm';

function JXLImage({ src }: { src: string }) {
  const [imgSrc, setImgSrc] = useState('');

  useEffect(() => {
    (async () => {
      await init();
      const res = await fetch(src);
      const jxlData = new Uint8Array(await res.arrayBuffer());
      const pngData = decode_jxl_to_png(jxlData);
      const blob = new Blob([pngData], { type: 'image/png' });
      setImgSrc(URL.createObjectURL(blob));
    })();
  }, [src]);

  return imgSrc ? <img src={imgSrc} /> : <div>Loading...</div>;
}
```

## API

### `decode_jxl_to_png(jxl_data: Uint8Array): Uint8Array`

Decode JXL to PNG. Returns APNG for animated images.

**Throws:** Error message if decoding fails.

### `get_jxl_info(jxl_data: Uint8Array): ImageInfo`

Get image metadata without full decode.

Returns:
- `width: number`
- `height: number`
- `num_frames: number`
- `has_alpha: boolean`

### `init_panic_hook(): void`

Enable better error messages in console. Call once at startup.

## Demo

```bash
./build.sh
cd demo
python3 -m http.server 8000
```

Open http://localhost:8000

Features:
- Multi-file drag & drop
- Per-image stats (dimensions, sizes, decode time)
- Download as PNG
- Remove/clear images

Test images: https://github.com/libjxl/conformance

## Performance

Decode times (approximate):
- <1MP: 10-50ms
- 1-5MP: 50-200ms
- \>5MP: 200-1000ms

Based on jxl-rs with SIMD optimizations (compiled out for WASM).

## Browser Support

Requires WebAssembly:
- Chrome 57+
- Firefox 52+
- Safari 11+
- Edge 79+

## Limitations

- Single-threaded (no parallel decoding)
- No WASM SIMD support yet
- Browser memory constraints apply

## License

BSD-3-Clause
