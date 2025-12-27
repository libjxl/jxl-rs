# Chromium Integration Samples

This folder contains sample code for integrating jxl-rs into Chromium's image decoding infrastructure.

**NOTE:** These files are examples only. The actual integration should be done within the Chromium repository itself.

## Files

### Rust CXX Wrappers

- `jxl_wrapper.rs` - CXX bridge wrapper for JXL decoding
- `jpeg_wrapper.rs` - CXX bridge wrapper for JPEG decoding

These files use the `cxx` crate to expose Rust APIs to C++. They should be placed in a Chromium-side crate that depends on `jxl`.

### C++ Decoder

- `jpeg_image_decoder.h` - Chromium ImageDecoder header for JPEG
- `jpeg_image_decoder.cc` - Chromium ImageDecoder implementation

These implement Blink's `ImageDecoder` interface using the Rust JPEG decoder.

## Dependencies

The Rust wrapper requires:
```toml
[dependencies]
cxx = "1.0"
jxl = { path = "...", features = ["jpeg-reconstruction"] }

[build-dependencies]
cxx-build = "1.0"
```

## jxl-rs JPEG API

The JPEG decoder exposes these public types from `jxl::jpeg`:

- `JpegDecoder` - Main decoder struct
- `JpegDecodedImage` - Decoded pixel data (f32 in 0.0-1.0 range)
- `JpegMetadata` - ICC profile, EXIF, XMP extraction

Example usage:
```rust
use jxl::jpeg::{JpegDecoder, JpegDecodedImage, JpegMetadata};

let decoder = JpegDecoder::new(&jpeg_data)?;
let metadata = decoder.extract_metadata();
let image = decoder.decode(&jpeg_data)?;

// image.pixels contains RGB f32 data
// image.width, image.height, image.num_components
```
