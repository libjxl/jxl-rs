#!/bin/bash -eu
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

cd $SRC/jxl-rs/jxl

# Build fuzz targets with cargo-fuzz
cargo fuzz build -O --debug-assertions

# Copy fuzz targets to output directory
FUZZ_TARGET_OUTPUT_DIR=fuzz/target/x86_64-unknown-linux-gnu/release
for f in fuzz/fuzz_targets/*.rs; do
    FUZZ_TARGET_NAME=$(basename "${f%.*}")
    if [ -f "$FUZZ_TARGET_OUTPUT_DIR/$FUZZ_TARGET_NAME" ]; then
        cp "$FUZZ_TARGET_OUTPUT_DIR/$FUZZ_TARGET_NAME" "$OUT/"
    fi
done

# Create seed corpus from test JXL files
# Both decode and decode_header targets benefit from real JXL samples
if [ -d "resources/test" ]; then
    zip -j "$OUT/decode_seed_corpus.zip" resources/test/*.jxl resources/test/conformance_test_images/*.jxl || true
    zip -j "$OUT/decode_header_seed_corpus.zip" resources/test/*.jxl resources/test/conformance_test_images/*.jxl || true
fi

# Also copy any manually curated corpus if available
if [ -d "fuzz/corpus" ]; then
    for target in fuzz/fuzz_targets/*.rs; do
        TARGET_NAME=$(basename "${target%.*}")
        if [ -d "fuzz/corpus/$TARGET_NAME" ]; then
            # Append to existing seed corpus or create new one
            zip -j "$OUT/${TARGET_NAME}_seed_corpus.zip" fuzz/corpus/"$TARGET_NAME"/* || true
        fi
    done
fi
