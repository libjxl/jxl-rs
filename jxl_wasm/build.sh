#!/bin/bash
set -e

echo "🔨 Building JXL WASM decoder..."

# Build the WASM module
wasm-pack build --target web --release

echo "✅ WASM module built successfully!"
echo ""
echo "📦 Copying to demo directory..."

# Copy to demo directory
mkdir -p demo/pkg
cp -r pkg/* demo/pkg/

echo "✅ Demo ready!"
echo ""
echo "🚀 To run the demo:"
echo "   cd demo"
echo "   python3 -m http.server 8000"
echo ""
echo "Then open http://localhost:8000 in your browser"
