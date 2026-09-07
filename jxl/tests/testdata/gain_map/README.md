# Gain-map reader fixtures

These small fixtures are used by the `JxlGainMapBundle` API tests. The
reproducible `generate_fixtures.py` script constructs the metadata, structured
linear-sRGB field, and 536-byte ICC profile from their field values. It invokes
`cjxl` for the lossless 2x2 RGB gain-map image, the standalone
`compress_icc.cc` helper for JPEG XL ICC coding, and `brotli` for the
compressed outer-box case. The helper uses libjxl's internal `WriteICC` API and
explicitly pads the stream to a byte boundary after writing the profile.

The metadata values exercise three channels and unequal denominators. The
Python serializer replaces the original libultrahdr v2 generation for this
small fixture set; its bytes were checked against libultrahdr commit
`418b6b361e252a91c435a56cf386afb37d7d1c9d`, without copying that dependency or
requiring it to build the fixtures. The ICC serializer mirrors
`JxlColorEncoding::srgb(false).maybe_create_profile()` from jxl-rs commit
`5236e98`; its output was compared byte-for-byte with `srgb.icc`.
`want_icc.jhgm` is a manually assembled envelope whose field retention is
intentional: its structured color field requests the alternate ICC field.

`synthetic-container.jxl` is an outer container with a finite trailing `jhgm`
box. `synthetic-container-brob.jxl` has the same payload in a finite Brotli
compressed `brob` box. `embedded-container.jhgm` is a synthetic permissive
byte-preservation fixture. It does not claim normative JHGM or ISO conformance
for an embedded container, and does not model observed real-world producer
output.

No fixture contains applied gain-map rendering. Tests verify envelope fields,
color/ICC decoding, nested 8-bit RGB image dimensions and exact samples,
compressed-box decompression, and malformed bounds only.

## Regeneration

The normal Rust build does not need these tools. Regeneration needs a configured
libjxl source/build pair with `libjxl-internal.a` and `libjxl_cms.a`, plus
`cjxl`, `brotli`, a C++17 compiler, and the dependency flags reported by
`pkg-config`. The source and build paths below are placeholders.

Byte identity was verified with libjxl source commit
`7f911c751ac6573f36beedc09ed5b01303565e49` (version 0.13.0), `cjxl` 0.12.0,
and Brotli 1.2.0. A different libjxl or tool version can legitimately produce
different compressed bytes.

```sh
FIXTURE_DIR=/path/to/jxl-rs/jxl/tests/testdata/gain_map
JXL_SOURCE=/path/to/libjxl
JXL_BUILD=/path/to/libjxl-build
SCRATCH=$(mktemp -d)

${CXX:-c++} -std=c++17 -fno-rtti -fno-exceptions \
  -I"$JXL_SOURCE" -I"$JXL_BUILD/lib/include" \
  "$FIXTURE_DIR/compress_icc.cc" \
  "$JXL_BUILD/lib/libjxl-internal.a" \
  "$JXL_BUILD/lib/libjxl_cms.a" \
  $(pkg-config --cflags --libs libhwy libbrotlienc libbrotlidec lcms2) \
  -o "$SCRATCH/compress_icc"

python3 "$FIXTURE_DIR/generate_fixtures.py" \
  --output "$SCRATCH/fixtures" \
  --compressor "$SCRATCH/compress_icc"
```

Compare every generated file with the checked-in set before replacing any
fixture bytes:

```sh
for name in combined.jhgm embedded-container.jhgm icc.jhgm srgb.icc \
    structured.jhgm synthetic-container-brob.jxl synthetic-container.jxl \
    synthetic.jxl want_icc.jhgm; do
  cmp "$SCRATCH/fixtures/$name" "$FIXTURE_DIR/$name"
done
```

The command is intentionally separate from the normal test build because the
ICC helper includes libjxl internal headers. A libjxl version change may alter
the JPEG XL ICC bitstream or the `cjxl` codestream; review such byte changes
before copying the scratch output into this directory.
