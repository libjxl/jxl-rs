#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Regenerate the small gain-map reader fixtures.

The script deliberately builds the metadata and ICC profile from their field
values. It only uses cjxl for the nested lossless image, the standalone
compress_icc helper for JPEG XL ICC coding, and the Brotli command line tool
for the optional compressed outer box.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path


PIXELS = bytes((0, 64, 128, 255, 254, 253, 32, 96, 160, 200, 100, 50))
FIXTURE_NAMES = (
    "combined.jhgm",
    "embedded-container.jhgm",
    "icc.jhgm",
    "srgb.icc",
    "structured.jhgm",
    "synthetic-container-brob.jxl",
    "synthetic-container.jxl",
    "synthetic.jxl",
    "want_icc.jhgm",
)


def u16(value: int) -> bytes:
    return struct.pack(">H", value)


def u32(value: int) -> bytes:
    return struct.pack(">I", value)


def s32(value: int) -> bytes:
    return struct.pack(">i", value)


def f32(value: float) -> float:
    """Round a Python float to the value Rust would store in an f32."""

    return struct.unpack(">f", struct.pack(">f", value))[0]


def round_away_from_zero(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def s15_fixed_16(value: float) -> bytes:
    value = f32(value)
    scaled = f32(value * 65536.0)
    return s32(round_away_from_zero(scaled))


def matrix_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[row][k] * b[k][column] for k in range(3)) for column in range(3)]
        for row in range(3)
    ]


def matrix_vector_multiply(a: list[list[float]], v: list[float]) -> list[float]:
    return [sum(a[row][column] * v[column] for column in range(3)) for row in range(3)]


def inverse_3x3(m: list[list[float]]) -> list[list[float]]:
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return [
        [
            (e * i - f * h) / determinant,
            (c * h - b * i) / determinant,
            (b * f - c * e) / determinant,
        ],
        [
            (f * g - d * i) / determinant,
            (a * i - c * g) / determinant,
            (c * d - a * f) / determinant,
        ],
        [
            (d * h - e * g) / determinant,
            (b * g - a * h) / determinant,
            (a * e - b * d) / determinant,
        ],
    ]


K_BRADFORD = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
]
K_BRADFORD_INV = [
    [0.9869929, -0.1470543, 0.1599627],
    [0.4323053, 0.5183603, 0.0492912],
    [-0.0085287, 0.0400428, 0.9684867],
]


def primaries_to_xyz(
    rx: float,
    ry: float,
    gx: float,
    gy: float,
    bx: float,
    by: float,
    wx: float,
    wy: float,
) -> list[list[float]]:
    primaries = [
        [rx, gx, bx],
        [ry, gy, by],
        [1.0 - rx - ry, 1.0 - gx - gy, 1.0 - bx - by],
    ]
    white = [wx / wy, 1.0, (1.0 - wx - wy) / wy]
    scales = matrix_vector_multiply(inverse_3x3(primaries), white)
    diagonal = [
        [scales[0], 0.0, 0.0],
        [0.0, scales[1], 0.0],
        [0.0, 0.0, scales[2]],
    ]
    return matrix_multiply(primaries, diagonal)


def adapt_to_xyz_d50(wx: float, wy: float) -> list[list[float]]:
    white = [wx / wy, 1.0, (1.0 - wx - wy) / wy]
    d50 = [0.96422, 1.0, 0.82521]
    source_lms = matrix_vector_multiply(K_BRADFORD, white)
    d50_lms = matrix_vector_multiply(K_BRADFORD, d50)
    diagonal = [
        [d50_lms[0] / source_lms[0], 0.0, 0.0],
        [0.0, d50_lms[1] / source_lms[1], 0.0],
        [0.0, 0.0, d50_lms[2] / source_lms[2]],
    ]
    return matrix_multiply(K_BRADFORD_INV, matrix_multiply(diagonal, K_BRADFORD))


def mluc(text: str) -> bytes:
    encoded = text.encode("ascii")
    return (
        b"mluc"
        + u32(0)
        + u32(1)
        + u32(12)
        + b"enUS"
        + u32(len(encoded) * 2)
        + u32(28)
        + b"".join(bytes((0, char)) for char in encoded)
    )


def xyz(values: list[float]) -> bytes:
    return b"XYZ " + u32(0) + b"".join(s15_fixed_16(value) for value in values)


def sf32(matrix: list[list[float]]) -> bytes:
    return b"sf32" + u32(0) + b"".join(
        s15_fixed_16(value) for row in matrix for value in row
    )


def para(values: list[float]) -> bytes:
    return b"para" + u32(0) + u16(3) + u16(0) + b"".join(
        s15_fixed_16(value) for value in values
    )


def make_srgb_icc() -> bytes:
    """Mirror JxlColorEncoding::srgb(false).maybe_create_profile()."""

    header = bytearray(128)
    header[4:8] = b"jxl "
    header[8:12] = u32(0x04400000)
    header[12:16] = b"mntr"
    header[16:20] = b"RGB "
    header[20:24] = b"XYZ "
    header[24:36] = struct.pack(">6H", 2019, 12, 1, 0, 0, 0)
    header[36:40] = b"acsp"
    header[40:44] = b"APPL"
    header[64:68] = u32(1)  # Relative rendering intent.
    header[68:72] = u32(0x0000F6D6)
    header[72:76] = u32(0x00010000)
    header[76:80] = u32(0x0000D32D)
    header[80:84] = b"jxl "

    tags_data = bytearray()
    tags: list[tuple[bytes, int, int]] = []

    def add_tag(signature: bytes, data: bytes) -> None:
        offset = len(tags_data)
        tags_data.extend(data)
        tags_data.extend(b"\0" * (-len(tags_data) % 4))
        tags.append((signature, offset, len(data)))

    add_tag(b"desc", mluc("RGB_D65_SRG_Rel_SRG"))
    add_tag(b"cprt", mluc("CC0"))
    add_tag(b"wtpt", xyz([f32(0.964203), 1.0, f32(0.824905)]))
    chad = adapt_to_xyz_d50(f32(0.3127), f32(0.3290))
    add_tag(b"chad", sf32([[f32(value) for value in row] for row in chad]))

    tags_data.extend(b"cicp" + u32(0) + bytes((1, 13, 0, 1)))
    tags.append((b"cicp", len(tags_data) - 12, 12))

    primaries = (
        (f32(0.6399987), f32(0.33001015)),
        (f32(0.3000038), f32(0.60000336)),
        (f32(0.15000205), f32(0.059997204)),
    )
    native = primaries_to_xyz(
        *(value for pair in primaries for value in pair), f32(0.3127), f32(0.3290)
    )
    adaptation = adapt_to_xyz_d50(f32(0.3127), f32(0.3290))
    matrix = matrix_multiply(adaptation, native)
    matrix = [[f32(value) for value in row] for row in matrix]
    add_tag(b"rXYZ", xyz([matrix[0][0], matrix[1][0], matrix[2][0]]))
    add_tag(b"gXYZ", xyz([matrix[0][1], matrix[1][1], matrix[2][1]]))
    add_tag(b"bXYZ", xyz([matrix[0][2], matrix[1][2], matrix[2][2]]))

    trc = para([2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045])
    add_tag(b"rTRC", trc)
    tags.append((b"gTRC", tags[-1][1], tags[-1][2]))
    tags.append((b"bTRC", tags[-1][1], tags[-1][2]))

    table = bytearray(u32(len(tags)))
    table_size = 4 + len(tags) * 12
    for signature, offset, size in tags:
        table.extend(signature + u32(len(header) + table_size + offset) + u32(size))

    profile = bytearray(header + table + tags_data)
    profile[:4] = u32(len(profile))
    profile_for_checksum = bytearray(profile)
    profile_for_checksum[44:48] = b"\0" * 4
    profile_for_checksum[64:68] = b"\0" * 4
    profile_for_checksum[84:100] = b"\0" * 16
    profile[84:100] = hashlib.md5(profile_for_checksum).digest()
    return bytes(profile)


def make_metadata() -> bytes:
    """Encode the fractional metadata values used by the original probe."""

    channel_values = (
        (-1, 3, 2, 7, 1, 2, 1, 5, 2, 9),
        (-2, 5, 3, 8, 2, 3, 2, 7, 3, 10),
        (-3, 7, 4, 9, 3, 4, 3, 8, 4, 11),
    )
    result = bytearray(u16(0) + u16(0) + bytes((0x80,)))
    result.extend(u32(1) + u32(2) + u32(3) + u32(5))
    for values in channel_values:
        (
            minimum_n,
            minimum_d,
            maximum_n,
            maximum_d,
            gamma_n,
            gamma_d,
            base_n,
            base_d,
            alternate_n,
            alternate_d,
        ) = values
        result.extend(
            s32(minimum_n)
            + u32(minimum_d)
            + s32(maximum_n)
            + u32(maximum_d)
            + u32(gamma_n)
            + u32(gamma_d)
            + s32(base_n)
            + u32(base_d)
            + s32(alternate_n)
            + u32(alternate_d)
        )
    assert len(result) == 141
    assert result[4] == 0x80
    return bytes(result)


def structured_linear_srgb() -> bytes:
    """Write the 17-bit linear-sRGB ColorEncoding using JPEG XL bit order."""

    bits: list[int] = []

    def add(value: int, count: int) -> None:
        bits.extend((value >> bit) & 1 for bit in range(count))

    # all_default=false, want_icc=false, RGB, D65, sRGB primaries,
    # no gamma, Linear transfer function, Relative intent.
    add(0, 1)
    add(0, 1)
    add(0, 2)  # RGB enum value 0.
    add(1, 2)  # D65 enum value 1.
    add(1, 2)  # sRGB primaries enum value 1.
    add(0, 1)  # have_gamma=false.
    add(2, 2)  # transfer-function selector for values >= 2.
    add(6, 4)  # Linear value 8, represented as 2 + 6.
    add(1, 2)  # Relative intent enum value 1.

    output = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        output[index // 8] |= bit << (index % 8)
    return bytes(output)


def raw_bundle(metadata: bytes, color: bytes, compressed_icc: bytes, gain_map: bytes) -> bytes:
    if (
        len(metadata) > 0xFFFF
        or len(color) > 0xFF
        or len(compressed_icc) > 0xFFFFFFFF
    ):
        raise ValueError("fixture field is too large")
    return (
        bytes((0,))
        + u16(len(metadata))
        + metadata
        + bytes((len(color),))
        + color
        + u32(len(compressed_icc))
        + compressed_icc
        + gain_map
    )


def box(kind: bytes, payload: bytes) -> bytes:
    return u32(8 + len(payload)) + kind + payload


def make_pixel_container(naked_jxl: bytes) -> bytes:
    signature = b"\0\0\0\x0cJXL \r\n\x87\n"
    file_type = u32(20) + b"ftypjxl " + u32(0) + b"jxl "
    return signature + file_type + box(b"jxlc", naked_jxl)


def run(command: list[str], *, input_data: bytes | None = None) -> None:
    subprocess.run(command, input=input_data, check=True)


def find_tool(path: str | None, name: str) -> str:
    resolved = path or shutil.which(name)
    if not resolved:
        raise SystemExit(f"could not find {name}; pass --{name} PATH")
    return resolved


def generate(output: Path, cjxl: str, compressor: str, brotli: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    metadata = make_metadata()
    profile = make_srgb_icc()
    structured_color = structured_linear_srgb()

    with tempfile.TemporaryDirectory(prefix="jxl-gain-map-") as temporary:
        temporary_path = Path(temporary)
        ppm = temporary_path / "synthetic.ppm"
        naked_jxl = temporary_path / "synthetic.jxl"
        compressed_icc = temporary_path / "icc.compressed"
        ppm.write_bytes(b"P6\n2 2\n255\n" + PIXELS)
        run([cjxl, "-d", "0", "-e", "3", str(ppm), str(naked_jxl)])
        (temporary_path / "srgb.icc").write_bytes(profile)
        run([compressor, str(temporary_path / "srgb.icc"), str(compressed_icc)])

        naked = naked_jxl.read_bytes()
        compressed = compressed_icc.read_bytes()
        pixel_container = make_pixel_container(naked)
        combined = raw_bundle(metadata, structured_color, compressed, naked)
        bundles = {
            "structured.jhgm": raw_bundle(metadata, structured_color, b"", naked),
            "icc.jhgm": raw_bundle(metadata, b"", compressed, naked),
            "combined.jhgm": combined,
            "embedded-container.jhgm": raw_bundle(
                metadata, structured_color, compressed, pixel_container
            ),
            "want_icc.jhgm": raw_bundle(metadata, b"\x02", compressed, naked),
        }
        files = {
            **bundles,
            "srgb.icc": profile,
            "synthetic.jxl": naked,
            "synthetic-container.jxl": pixel_container + box(b"jhgm", combined),
        }
        brob_payload = b"jhgm" + subprocess.run(
            [brotli, "-q", "4", "-c"], input=combined, check=True, stdout=subprocess.PIPE
        ).stdout
        files["synthetic-container-brob.jxl"] = pixel_container + box(
            b"brob", brob_payload
        )

    for name, data in files.items():
        (output / name).write_bytes(data)
    missing = set(FIXTURE_NAMES) - set(files)
    assert not missing, f"generator omitted fixtures: {sorted(missing)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="directory for generated fixtures"
    )
    parser.add_argument("--cjxl", help="path to cjxl (default: search PATH)")
    parser.add_argument(
        "--compressor",
        required=True,
        help="path to the compiled compress_icc helper",
    )
    parser.add_argument("--brotli", help="path to brotli (default: search PATH)")
    args = parser.parse_args()
    generate(
        args.output,
        find_tool(args.cjxl, "cjxl"),
        str(Path(args.compressor).resolve()),
        find_tool(args.brotli, "brotli"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
