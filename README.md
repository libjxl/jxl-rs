# JPEG XL in Rust

This is a work-in-progress reimplementation of a JPEG XL decoder in Rust, aiming to be conforming, safe, and fast.

We strive to decode all conformant JPEG XL bitstreams. If you find an image that can be decoded with the reference 
implementation `djxl` (from [`libjxl`](https://github.com/libjxl/libjxl)) but not with `jxl-rs`, 
please report it by [opening an issue](https://github.com/libjxl/jxl-rs/issues/new).

For more information, including contributing instructions, refer to the [libjxl repository](https://github.com/libjxl/libjxl).