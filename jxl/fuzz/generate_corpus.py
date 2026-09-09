#!/usr/bin/env python3
"""Generates seed corpus for jxl-rs fuzz targets.

Generates tricky sizes (group boundaries, SIMD tails, extreme aspect ratios),
harvests test files from resources/test/, slices prefixes, and populates
jxl/fuzz/corpus/<target>/ directories.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FUZZ_DIR = Path(__file__).resolve().parent
REPO_ROOT = FUZZ_DIR.parent.parent
RESOURCES_TEST = REPO_ROOT / "jxl" / "resources" / "test"
CORPUS_BASE = FUZZ_DIR / "corpus"

TARGETS = [
    "decode",
    "decode_parallel",
    "decode_progressive",
    "decode_progressive_parallel",
    "decode_diff",
    "decode_header",
]

# Tricky sizes focusing on:
# - SIMD vector widths (1, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65)
# - Group boundary (256): 255, 256, 257
# - Multi-group boundary: 511, 512, 513
# - Extreme aspect ratios crossing group boundaries: (1, 257), (257, 1), (1, 1024), (1024, 1)
SIZES = [
    (1, 1),
    (3, 3),
    (7, 7),
    (8, 8),
    (9, 9),
    (15, 15),
    (16, 16),
    (17, 17),
    (31, 31),
    (32, 32),
    (33, 33),
    (63, 63),
    (64, 64),
    (65, 65),
    (255, 255),
    (256, 256),
    (257, 256),
    (256, 257),
    (257, 257),
    (511, 511),
    (512, 512),
    (513, 513),
    (1, 257),
    (257, 1),
    (1, 1024),
    (1024, 1),
]

def find_binary(names):
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main():
    magick = find_binary(["magick", "convert"])
    cjxl = find_binary(["cjxl"])
    
    print(f"Magick: {magick}")
    print(f"cjxl: {cjxl}")
    
    unique_files = {}  # sha256 -> bytes

    def add_data(data: bytes, desc: str):
        if not data:
            return False
        h = sha256_bytes(data)
        if h not in unique_files:
            unique_files[h] = data
            return True
        return False

    # 1. Harvest from resources/test
    if RESOURCES_TEST.exists():
        print("Harvesting test files from resources/test...")
        count = 0
        for f in RESOURCES_TEST.rglob("*.jxl"):
            try:
                # Limit initial seed files to <= 64KB for fast fuzzing execution
                if f.is_file() and f.stat().st_size <= 65536:
                    data = f.read_bytes()
                    if add_data(data, f.name):
                        count += 1
                        # Add prefixes for progressive/streaming exploration
                        for prefix_len in (64, 256, 1024, 4096):
                            if len(data) > prefix_len:
                                add_data(data[:prefix_len], f"{f.name}-p{prefix_len}")
            except Exception as e:
                print(f"Warning reading {f}: {e}", file=sys.stderr)
        print(f"Harvested {count} test files.")

    # 2. Generate synthetic images with tricky dimensions
    if magick and cjxl:
        print("Generating images with tricky dimensions...")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gen_count = 0
            for w, h in SIZES:
                for has_alpha in (False, True):
                    png_path = tmp / f"test_{w}x{h}_{'rgba' if has_alpha else 'rgb'}.png"
                    # Generate pattern PNG using magick
                    if has_alpha:
                        cmd = [magick, "-size", f"{w}x{h}", "plasma:fractal", "-alpha", "set", "-channel", "A", "-evaluate", "sine", "2", str(png_path)]
                    else:
                        cmd = [magick, "-size", f"{w}x{h}", "pattern:checkerboard", str(png_path)]
                    
                    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if res.returncode != 0 or not png_path.exists():
                        continue
                    
                    # Encode with cjxl in both VarDCT (lossy) and Modular (lossless)
                    # VarDCT mode (lossy)
                    jxl_vardct = tmp / f"test_{w}x{h}_{'rgba' if has_alpha else 'rgb'}_vardct.jxl"
                    res = subprocess.run([cjxl, str(png_path), str(jxl_vardct), "-d", "1.0", "-e", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if res.returncode == 0 and jxl_vardct.exists():
                        data = jxl_vardct.read_bytes()
                        if add_data(data, jxl_vardct.name):
                            gen_count += 1
                            for prefix_len in (64, 256, 1024):
                                if len(data) > prefix_len:
                                    add_data(data[:prefix_len], f"{jxl_vardct.name}-p{prefix_len}")

                    # Modular mode (lossless)
                    jxl_modular = tmp / f"test_{w}x{h}_{'rgba' if has_alpha else 'rgb'}_modular.jxl"
                    res = subprocess.run([cjxl, str(png_path), str(jxl_modular), "-m", "-d", "0", "-e", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if res.returncode == 0 and jxl_modular.exists():
                        data = jxl_modular.read_bytes()
                        if add_data(data, jxl_modular.name):
                            gen_count += 1
                            for prefix_len in (64, 256, 1024):
                                if len(data) > prefix_len:
                                    add_data(data[:prefix_len], f"{jxl_modular.name}-p{prefix_len}")
            print(f"Generated {gen_count} tricky-sized JXL files.")

    print(f"Total unique seeds gathered: {len(unique_files)}")

    # 3. Populate target corpora
    for target in TARGETS:
        target_dir = CORPUS_BASE / target
        target_dir.mkdir(parents=True, exist_ok=True)
        added = 0
        for h, data in unique_files.items():
            dest = target_dir / h
            if not dest.exists():
                dest.write_bytes(data)
                added += 1
        print(f"Target '{target}': {len(list(target_dir.glob('*')))} files total ({added} newly added).")

if __name__ == "__main__":
    main()
