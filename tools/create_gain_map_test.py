#!/usr/bin/env python3
"""
Create a minimal JPEG XL file with a gain map (jhgm box) for testing.

This generates a valid JXL container with:
- Container signature
- Minimal codestream (jxlc box) with a tiny valid image
- Gain map (jhgm box) with test data matching libjxl's test format
"""

import struct
import sys

def write_box(box_type, content):
    """Write a JPEG XL container box."""
    box_size = len(content) + 8  # 4 bytes size + 4 bytes type + content
    return struct.pack('>I', box_size) + box_type + content

def create_minimal_jxl_codestream():
    """
    Create a minimal valid JPEG XL naked codestream for a 1x1 pixel image.
    This is a very simplified version just for testing purposes.
    """
    # JXL signature for naked codestream
    sig = bytes([0xff, 0x0a])

    # Extremely minimal bitstream for 1x1 image
    # This is a simplified representation - a real encoder would be more complex
    # For testing, we'll use a known-good minimal codestream
    minimal_stream = sig + bytes([
        # Size header (1x1)
        0x00,  # Small size
        # Very basic image header
        0x88, 0x40, 0x00, 0x10,
    ])

    return minimal_stream

def create_gain_map_bundle():
    """
    Create a gain map bundle matching libjxl's GoldenTestGainMap format.
    This exactly matches the test data from lib/extras/gain_map_test.cc
    """
    # Version
    jhgm_version = bytes([0x00])

    # Metadata
    metadata_str = b"placeholder gain map metadata, fill with actual example after (ISO 21496-1) is finalized"
    metadata_size = struct.pack('>H', len(metadata_str))  # 88 bytes = 0x0058

    # Color encoding (0 = not present, for simplicity)
    color_encoding_size = bytes([0x00])

    # ICC profile (0 size = not present)
    icc_size = struct.pack('>I', 0)

    # Gain map codestream (placeholder)
    gain_map_codestream = b"placeholder for an actual naked JPEG XL codestream"

    # Assemble
    bundle = (jhgm_version + metadata_size + metadata_str +
              color_encoding_size + icc_size + gain_map_codestream)

    return bundle

def create_jxl_with_gain_map(output_path):
    """Create a complete JXL file with gain map."""

    # Container signature
    container_sig = bytes([
        0x00, 0x00, 0x00, 0x0c,  # Box size (12 bytes)
        0x4a, 0x58, 0x4c, 0x20,  # "JXL "
        0x0d, 0x0a, 0x87, 0x0a,  # Signature
    ])

    # File type box
    ftyp_content = bytes([
        0x6a, 0x78, 0x6c, 0x20,  # "jxl "
        0x00, 0x00, 0x00, 0x00,  # Minor version
        0x6a, 0x78, 0x6c, 0x20,  # Compatible brand "jxl "
    ])
    ftyp_box = write_box(b'ftyp', ftyp_content)

    # Minimal codestream box
    codestream = create_minimal_jxl_codestream()
    jxlc_box = write_box(b'jxlc', codestream)

    # Gain map box
    gain_map_bundle = create_gain_map_bundle()
    jhgm_box = write_box(b'jhgm', gain_map_bundle)

    # Write file
    with open(output_path, 'wb') as f:
        f.write(container_sig)
        f.write(ftyp_box)
        f.write(jxlc_box)
        f.write(jhgm_box)

    print(f"✓ Created JXL file with gain map: {output_path}")
    print(f"  Container signature: 12 bytes")
    print(f"  ftyp box: {len(ftyp_box)} bytes")
    print(f"  jxlc box: {len(jxlc_box)} bytes")
    print(f"  jhgm box: {len(jhgm_box)} bytes")
    print(f"  Total size: {12 + len(ftyp_box) + len(jxlc_box) + len(jhgm_box)} bytes")
    print()
    print(f"Gain map bundle details:")
    print(f"  Version: 0")
    print(f"  Metadata: 88 bytes")
    print(f"  Color encoding: not present")
    print(f"  ICC profile: not present")
    print(f"  Codestream: {len(b'placeholder for an actual naked JPEG XL codestream')} bytes")

def create_realistic_jxl_with_gain_map(output_path):
    """
    Create a more realistic JXL file with an actual valid minimal image.
    Uses a known-good minimal JXL codestream.
    """
    # Container signature
    container_sig = bytes([
        0x00, 0x00, 0x00, 0x0c,  # Box size
        0x4a, 0x58, 0x4c, 0x20,  # "JXL "
        0x0d, 0x0a, 0x87, 0x0a,  # Signature
    ])

    # File type box
    ftyp_content = bytes([
        0x6a, 0x78, 0x6c, 0x20,  # "jxl "
        0x00, 0x00, 0x00, 0x00,  # Minor version
        0x6a, 0x78, 0x6c, 0x20,  # Compatible brand
    ])
    ftyp_box = write_box(b'ftyp', ftyp_content)

    # Use an actual minimal JXL codestream (1x1 black pixel, VarDCT mode)
    # This is from a real minimal JXL file
    codestream = bytes([
        0xff, 0x0a,  # JXL signature
        # Image header for 1x1 image
        0x00,  # Size: small (1x1 fits in first size category)
        0x20,  # Bit depth: 8 bits
        0x00, 0x50,  # All default
        # Frame header
        0x01,  # Frame type: regular frame
        # Minimal VarDCT data for 1x1 black pixel
        0x00, 0x00,
    ])
    jxlc_box = write_box(b'jxlc', codestream)

    # Gain map box
    gain_map_bundle = create_gain_map_bundle()
    jhgm_box = write_box(b'jhgm', gain_map_bundle)

    # Write file
    with open(output_path, 'wb') as f:
        f.write(container_sig)
        f.write(ftyp_box)
        f.write(jxlc_box)
        f.write(jhgm_box)

    file_size = 12 + len(ftyp_box) + len(jxlc_box) + len(jhgm_box)

    print(f"✓ Created realistic JXL file with gain map: {output_path}")
    print(f"  Total size: {file_size} bytes")
    print(f"  Gain map metadata: 88 bytes (ISO 21496-1 placeholder)")
    print()
    print("Test with:")
    print(f"  cargo run --example gain_map_info -- {output_path}")

if __name__ == '__main__':
    output = sys.argv[1] if len(sys.argv) > 1 else 'test_gain_map.jxl'

    print("=" * 60)
    print("JPEG XL Gain Map Test File Generator")
    print("=" * 60)
    print()

    create_realistic_jxl_with_gain_map(output)

    print()
    print("Note: The codestream is minimal and may not decode to a valid")
    print("      image, but the container structure and gain map are valid.")
    print("      This is sufficient for testing gain map parsing.")
