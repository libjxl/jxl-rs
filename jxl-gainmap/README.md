# `jxl-gainmap`

`jxl-gainmap` provides helpers for inspecting the payload of a JPEG XL `jhgm`
gain-map box. `JxlGainMap` borrows metadata and the nested image while decoding
the selected optional alternate color profile during parsing. Metadata
interpretation, nested image decoding, and gain-map rendering remain the
caller's responsibility.

```toml
[dependencies]
jxl = { version = "0.7.4", default-features = false }
jxl-gainmap = { version = "0.7.4", default-features = false }
```

Pass the uncompressed box payload, without the outer box header, to the
parser:

```rust
use jxl_gainmap::JxlGainMap;

fn inspect(payload: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let bundle = JxlGainMap::parse(payload)?;
    let alternate_profile = &bundle.color_profile;
    let embedded_jxl = bundle.gain_map;

    // Interpret the fields and decode embedded_jxl as appropriate.
    let _ = (alternate_profile, embedded_jxl);
    Ok(())
}
```

`parse()` validates and resolves the selected profile into `color_profile`.
`None` means that both alternate fields are absent, so the caller can use the
baseline profile. ICC bytes are owned by the returned value; metadata and the
nested image remain borrowed. A structured field uses `want_icc` to select the
structured encoding or ICC field. Malformed selected data or missing required
ICC returns an error, while unused ICC bytes are ignored when `want_icc=false`.
Only version zero is supported; unsupported versions return an error.

The core `jxl` decoder exposes captured auxiliary boxes through
`JxlAuxBoxType::HDR_GAIN_MAP`; if the enclosing `brob` box is Brotli-compressed,
enable `jxl`'s `brotli` feature and decompress it with `JxlAuxBox::data` first.

The included example accepts a finite, uncompressed `jhgm` box and decodes its
embedded image:

```sh
cargo run --release -p jxl-gainmap --example gain_map -- input.jxl
```

The crate keeps `jxl`'s default features disabled. To accelerate the example's
image decoding, add `--features jxl/all-simd` to the command.
