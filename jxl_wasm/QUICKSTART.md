# Quick Start Guide

Get started with the JXL WASM decoder in 3 steps:

## 1. Build the WASM Module

```bash
cd jxl_wasm
./build.sh
```

Or manually:
```bash
wasm-pack build --target web --release
```

## 2. Run the Demo

```bash
cd demo
python3 -m http.server 8000
```

Then open http://localhost:8000

## 3. Use in Your Project

### HTML + Vanilla JavaScript

```html
<!DOCTYPE html>
<html>
<head>
    <title>JXL Decoder</title>
</head>
<body>
    <input type="file" id="file" accept=".jxl">
    <img id="output">

    <script type="module">
        import init, { decode_jxl_to_png } from './pkg/jxl_wasm.js';

        await init();

        document.getElementById('file').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            const jxlData = new Uint8Array(await file.arrayBuffer());
            const pngData = decode_jxl_to_png(jxlData);
            const blob = new Blob([pngData], { type: 'image/png' });
            document.getElementById('output').src = URL.createObjectURL(blob);
        });
    </script>
</body>
</html>
```

### TypeScript + React

```tsx
import { useEffect, useState } from 'react';
import init, { decode_jxl_to_png } from './pkg/jxl_wasm';

export function JXLImage({ src }: { src: string }) {
    const [imgSrc, setImgSrc] = useState('');

    useEffect(() => {
        (async () => {
            await init();
            const response = await fetch(src);
            const jxlData = new Uint8Array(await response.arrayBuffer());
            const pngData = decode_jxl_to_png(jxlData);
            const blob = new Blob([pngData], { type: 'image/png' });
            setImgSrc(URL.createObjectURL(blob));
        })();
    }, [src]);

    return imgSrc ? <img src={imgSrc} /> : <div>Loading...</div>;
}
```

### Next.js 13+ (App Router)

```tsx
'use client';

import { useEffect, useState } from 'react';

export default function JXLViewer() {
    const [init, setInit] = useState<any>(null);

    useEffect(() => {
        import('./pkg/jxl_wasm').then(module => {
            module.default().then(() => setInit(() => module));
        });
    }, []);

    const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!init || !e.target.files?.[0]) return;

        const file = e.target.files[0];
        const jxlData = new Uint8Array(await file.arrayBuffer());
        const pngData = init.decode_jxl_to_png(jxlData);
        // Use pngData...
    };

    return <input type="file" onChange={handleFile} accept=".jxl" />;
}
```

## API Reference

### `decode_jxl_to_png(jxl_data: Uint8Array): Uint8Array`
Decode JXL to PNG/APNG bytes

### `get_jxl_info(jxl_data: Uint8Array): ImageInfo`
Get image dimensions and metadata

### `init_panic_hook(): void`
Enable better error messages (call once at startup)

## Performance Tips

1. **Initialize once**: Call `init()` once when your app loads
2. **Use Workers**: Decode large images in a Web Worker
3. **Cache results**: Store decoded PNGs if you'll reuse them
4. **Lazy load**: Only initialize when JXL decoding is needed

## Troubleshooting

**Module not found**: Make sure the `pkg` directory is in the correct location

**CORS errors**: Serve files through a web server, not file://

**Out of memory**: Large images may exceed browser memory limits

**Slow performance**: This is expected - decoding is CPU intensive

## Next Steps

- Read the full [README](README.md) for detailed documentation
- Check out [examples](demo/) for more use cases
- Visit [jxl-rs](https://github.com/libjxl/jxl-rs) for the underlying library
