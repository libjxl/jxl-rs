# API color profile test data

`encoded_srgb.icc` is the 228-byte `compressed_icc` field from
`jxl-gainmap/tests/testdata/gain_map/icc.jhgm`.

SHA-256: `7b8b9dfd46781cad1d4200bfae5908e398c8622faef0d14102c4e5be43cabd4c`

The decoded profile matches `JxlColorEncoding::srgb(false).maybe_create_profile()`;
its JPEG XL ICC stream was generated with libjxl 0.13.0.
