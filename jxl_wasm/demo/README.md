# Demo

Browser demo for the JXL WASM decoder.

## Run

```bash
python3 -m http.server 8000
```

Open http://localhost:8000

## Requirements

WASM module must be built first:
```bash
cd ..
./build.sh
```

This copies `pkg/` to `demo/pkg/`.

## Test Images

- https://github.com/libjxl/conformance
- https://jpegxl.info/
