#!/bin/bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

INPUT="$1"
OUTPUT_DIR="$2"

if [ -z "$INPUT" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <input_jxl> <output_dir>"
  exit 1
fi

echo "Building jxl_cli..."
cargo build --bin jxl_cli --release

BINARY="./target/release/jxl_cli"

if [ ! -f "$BINARY" ]; then
  echo "Error: Binary not found at $BINARY after build."
  exit 1
fi

if [ ! -f "$INPUT" ]; then
  echo "Error: Input file not found at $INPUT"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
# Clean up previous runs
rm -f "$OUTPUT_DIR"/*.jxl "$OUTPUT_DIR"/*.png "$OUTPUT_DIR"/*.mp4

# Get file size
if [[ "$OSTYPE" == "darwin"* ]]; then
  FILE_SIZE=$(stat -f%z "$INPUT")
else
  FILE_SIZE=$(stat -c%s "$INPUT")
fi

echo "Input file: $INPUT"
echo "File size: $FILE_SIZE bytes"
echo "Output directory: $OUTPUT_DIR"

for i in $(seq 5 5 100); do
  PERCENT=$i
  SIZE=$((FILE_SIZE * PERCENT / 100))
  if [ "$SIZE" -lt 1 ]; then SIZE=1; fi

  echo "Processing $PERCENT% ($SIZE bytes)..."

  TRUNCATED_FILE="$OUTPUT_DIR/truncated_$PERCENT.jxl"
  OUTPUT_IMG="$OUTPUT_DIR/frame_$(printf "%03d" $i).png"

  head -c "$SIZE" "$INPUT" >"$TRUNCATED_FILE"

  "$BINARY" "$TRUNCATED_FILE" "$OUTPUT_IMG" --allow-partial-files
done

echo "Creating video..."
VIDEO_OUTPUT="$OUTPUT_DIR/progressive.mp4"
ffmpeg -y -framerate 2 -pattern_type glob -i "$OUTPUT_DIR/frame_*.png" -vf "scale='2*trunc(min(3840,iw)/2)':-2" -c:v libx264 -pix_fmt yuv420p "$VIDEO_OUTPUT"
