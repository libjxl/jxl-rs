# `jxl-gainmap`

`jxl-gainmap` provides helpers for inspecting the payload of a JPEG XL `jhgm`
gain-map box. It parses the bundle fields and decodes the optional alternate
color encoding and ICC profile. Metadata interpretation, nested image decoding,
and gain-map rendering remain the caller's responsibility.

```toml
[dependencies]
jxl = { version = "0.7.4", default-features = false }
jxl-gainmap = { version = "0.7.4", default-features = false }
```

Pass the uncompressed box payload, without the outer box header, to the
parser:

```rust
use jxl_gainmap::JxlGainMapBundle;

fn inspect(payload: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let bundle = JxlGainMapBundle::parse(payload)?;
    let alternate_color = bundle.decode_color_encoding()?;
    let alternate_icc = bundle.decode_alternate_icc()?;
    let embedded_jxl = bundle.gain_map;

    // Interpret the fields and decode embedded_jxl as appropriate.
    let _ = (alternate_color, alternate_icc, embedded_jxl);
    Ok(())
}
```

The core `jxl` decoder exposes captured auxiliary boxes through
`JxlAuxBoxType::GAIN_MAP`. Enable the `brotli` feature on `jxl-gainmap` when
the input can contain a Brotli-compressed `brob` box; then decompress the
captured box with `JxlAuxBox::data` before parsing it.

The included example accepts a finite, uncompressed `jhgm` box and decodes its
embedded image:

```sh
cargo run --release -p jxl-gainmap --example gain_map -- input.jxl
```

The crate defaults to `all-simd` and forwards the individual SIMD feature
flags supported by `jxl`. Disable default features to select them explicitly.
