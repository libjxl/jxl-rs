#!/bin/bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Inspects metadata in a JXL file using jxl_cli and exiftool.

set -e

INPUT="$1"
OUTPUT_DIR="$2"

if [ -z "$INPUT" ]; then
  echo "Usage: $0 <input_jxl> [output_dir]"
  echo ""
  echo "Extracts and inspects metadata from a JXL file."
  echo "If output_dir is not specified, a temporary directory is used."
  exit 1
fi

if [ ! -f "$INPUT" ]; then
  echo "Error: Input file not found at $INPUT"
  exit 1
fi

echo "Building jxl_cli..."
cargo build --bin jxl_cli --release

BINARY="./target/release/jxl_cli"

if [ ! -f "$BINARY" ]; then
  echo "Error: Binary not found at $BINARY after build."
  exit 1
fi

CLEANUP_DIR=""
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR=$(mktemp -d)
  CLEANUP_DIR="$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "=== jxl_cli --info ==="
"$BINARY" "$INPUT" --info --metadata-out "$OUTPUT_DIR"

echo ""
echo "=== exiftool (JXL file) ==="
exiftool "$INPUT"

if [ -f "$OUTPUT_DIR/metadata_exif.exif" ]; then
  echo ""
  echo "=== exiftool (extracted EXIF) ==="
  exiftool "$OUTPUT_DIR/metadata_exif.exif"
fi

if [ -n "$CLEANUP_DIR" ]; then
  rm -rf "$CLEANUP_DIR"
fi
