#!/bin/bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
#
# Streaming progressive decode demo: uses jxl_cli's --progressive-step-size
# to produce all preview frames in a single decoder run, simulating streaming decode.

set -e

INPUT="$1"
OUTPUT_DIR="$2"
STEP_SIZE="${3:-5%}"  # Default to 5% if not specified

if [ -z "$INPUT" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <input_jxl> <output_dir> [step_size]"
  echo ""
  echo "Arguments:"
  echo "  input_jxl   - Path to input JPEG XL file"
  echo "  output_dir  - Directory for output frames and video"
  echo "  step_size   - Progressive step size (default: 5%)"
  echo "                Can be absolute (e.g. '1000' for bytes) or percentage (e.g. '5%')"
  echo ""
  echo "Example:"
  echo "  $0 image.jxl output/ 5%"
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
rm -f "$OUTPUT_DIR"/*.png "$OUTPUT_DIR"/*.mp4

# Get file size
if [[ "$OSTYPE" == "darwin"* ]]; then
  FILE_SIZE=$(stat -f%z "$INPUT")
else
  FILE_SIZE=$(stat -c%s "$INPUT")
fi

echo "Input file: $INPUT"
echo "File size: $FILE_SIZE bytes"
echo "Output directory: $OUTPUT_DIR"
echo "Step size: $STEP_SIZE"
echo ""

# Use a temporary output path - the actual frames will be named with frame numbers
TEMP_OUTPUT="$OUTPUT_DIR/temp.png"

echo "Running streaming progressive decode..."
"$BINARY" "$INPUT" "$TEMP_OUTPUT" --progressive-step-size "$STEP_SIZE" --allow-partial-files

# Remove the temp output if it exists (jxl_cli creates intermediate frames, not this one)
rm -f "$TEMP_OUTPUT"

echo ""
echo "Creating video from frames..."
VIDEO_OUTPUT="$OUTPUT_DIR/progressive_streaming.mp4"

# Count frames
FRAME_COUNT=$(ls -1 "$OUTPUT_DIR"/temp-*.png 2>/dev/null | wc -l)

if [ "$FRAME_COUNT" -eq 0 ]; then
  echo "Error: No frames were generated!"
  exit 1
fi

echo "Found $FRAME_COUNT frames"

# Flatten frames with checkerboard background (for alpha transparency)
echo "Flattening frames with checkerboard background..."
FLATTENED_DIR="$OUTPUT_DIR/flattened"
mkdir -p "$FLATTENED_DIR"

for frame in "$OUTPUT_DIR"/temp-*.png; do
  if [ -f "$frame" ]; then
    basename=$(basename "$frame")
    output="$FLATTENED_DIR/$basename"
    
    # Get image dimensions
    dimensions=$(identify -format "%wx%h" "$frame")
    
    # Use ImageMagick to composite frame over checkerboard pattern
    # Create a 32x32 checkerboard tile (16x16 pixel squares, light gray and white)
    # Background first, then frame on top with Over composite
    convert -size 16x16 xc:white xc:'#d0d0d0' +append \
            -write mpr:row1 +delete \
            -size 16x16 xc:'#d0d0d0' xc:white +append \
            -write mpr:row2 +delete \
            mpr:row1 mpr:row2 -append -write mpr:tile +delete \
            -size "$dimensions" tile:mpr:tile \
            "$frame" -compose Over -composite "$output"
  fi
done

echo "Flattened frames saved to: $FLATTENED_DIR/"

# Create video from flattened frames (2 fps to see progression clearly)
ffmpeg -y -framerate 2 -pattern_type glob -i "$FLATTENED_DIR/temp-*.png" \
  -vf "scale='2*trunc(min(3840,iw)/2)':-2" \
  -c:v libx264 -pix_fmt yuv420p "$VIDEO_OUTPUT"

echo ""
echo "Done!"
echo "Frames saved to: $OUTPUT_DIR/temp-*.png"
echo "Video saved to: $VIDEO_OUTPUT"
echo ""
echo "To view the video:"
echo "  ffplay $VIDEO_OUTPUT"
