#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import argparse
import glob
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Progressive decoding demo video generator for JPEG XL"
    )
    parser.add_argument("input", type=Path, help="Input JXL image file")
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for generated frames and videos"
    )

    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument(
        "--step-fraction",
        type=float,
        default=0.01,
        help="Interval as a fraction of file size (default: 0.01, representing 1%%)",
    )
    step_group.add_argument(
        "--step",
        type=int,
        help="Interval in bytes (e.g. 10000)",
    )

    parser.add_argument(
        "--mode",
        "--modes",
        nargs="+",
        choices=["linear", "quadratic", "both"],
        default=["linear", "quadratic"],
        dest="modes",
        help="Timing mode(s) for intervals (default: linear quadratic)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Duration of progressive portion in seconds (default: 8.0)",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=2.0,
        help="Duration of final frame hold in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--build",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build jxl_cli before running (default: True)",
    )
    parser.add_argument(
        "--annotate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate video with file size subtitles (default: True)",
    )

    return parser.parse_args()


def fmt_size(b: int) -> str:
    if b < 1024:
        return f"{b} B"
    elif b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    else:
        return f"{b / (1024 * 1024):.2f} MB"


def fmt_srt_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis -= 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def get_png_dimensions(png_path: Path):
    with open(png_path, "rb") as f:
        header = f.read(24)
        if len(header) >= 24 and header[:8] == bytes(
            [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        ):
            w, h = struct.unpack(">II", header[16:24])
            return w, h
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        str(png_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    w, h = out.split("x")
    return int(w), int(h)


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    binary = repo_root / "target" / "release" / "jxl_cli"

    input_path = args.input.resolve()
    if not input_path.is_file():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    file_size = input_path.stat().st_size
    print(f"Input file: {input_path}")
    print(f"File size: {file_size} bytes")

    if args.build:
        print("Building jxl_cli...")
        subprocess.check_call(
            ["cargo", "build", "--bin", "jxl_cli", "--release"], cwd=repo_root
        )

    if not binary.is_file():
        print(f"Error: Binary not found at {binary} after build.", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous runs in output_dir
    for ext in ("*.jxl", "*.png", "*.mp4", "*.txt", "*.srt"):
        for f in output_dir.glob(ext):
            try:
                f.unlink()
            except OSError:
                pass

    if args.step is not None:
        chunk_size = max(1, args.step)
    else:
        chunk_size = max(1, int(file_size * args.step_fraction))

    print(f"Render interval: {chunk_size} bytes")
    print(f"Output directory: {output_dir}")

    frame_output = output_dir / "frame.png"
    cmd = [
        str(binary),
        str(input_path),
        str(frame_output),
        "--render-interval",
        str(chunk_size),
    ]
    subprocess.check_call(cmd)

    if not frame_output.is_file():
        print(f"Error: Failed to produce final image {frame_output}", file=sys.stderr)
        sys.exit(1)

    width, height = get_png_dimensions(frame_output)
    print(f"Image dimensions: {width}x{height}")

    black_frame = output_dir / "black.png"
    print("Generating initial black frame...")
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}",
            "-frames:v",
            "1",
            "-update",
            "1",
            str(black_frame),
        ]
    )

    # Gather partial frames
    partial_files = sorted(
        output_dir.glob("frame.partial_*.png"),
        key=lambda p: (
            int(re.search(r"frame\.partial_(\d+)\.png$", p.name).group(1))
            if re.search(r"frame\.partial_(\d+)\.png$", p.name)
            else 0
        ),
    )

    if not partial_files:
        print("Error: No partial render frames found!", file=sys.stderr)
        sys.exit(1)

    items = [("black.png", 0)]
    for p in partial_files:
        byte_idx = int(re.search(r"frame\.partial_(\d+)\.png$", p.name).group(1))
        items.append((p.name, byte_idx))

    total_bytes = max(file_size, items[-1][1])

    modes = []
    for m in args.modes:
        if m == "both":
            modes.extend(["linear", "quadratic"])
        else:
            modes.append(m)
    modes = list(dict.fromkeys(modes))

    fps = 25
    filter_base = (
        "scale='2*trunc(min(3840,iw)/2)':-2,format=rgba,split[v][alpha];"
        f"[v]drawbox=c=black:t=fill[bg];[bg][alpha]overlay,fps={fps}"
    )
    sub_style = "force_style='FontSize=22,PrimaryColour=&HFFFFFF,BackColour=&H80000000,BorderStyle=4'"

    for mode in modes:
        print(f"Generating manifests for {mode} mode...")
        start_times = []
        for name, b in items:
            frac = min(1.0, b / total_bytes) if total_bytes > 0 else 1.0
            if mode == "linear":
                t = args.duration * frac
            else:
                t = args.duration * math.sqrt(frac)
            start_times.append(t)

        start_times[0] = 0.0

        concat_lines = ["ffconcat version 1.0"]
        for i in range(len(items)):
            name, b = items[i]
            t_start = start_times[i]
            if i + 1 < len(items):
                t_end = start_times[i + 1]
            else:
                t_end = start_times[-1] + args.hold

            d = max(0.001, t_end - t_start)
            concat_lines.append(f"file {name}")
            concat_lines.append(f"duration {d:.4f}")

        concat_lines.append(f"file {items[-1][0]}")

        concat_file = output_dir / f"concat_{mode}.txt"
        with open(concat_file, "w") as f:
            f.write("\n".join(concat_lines) + "\n")

        # Generate continuous ticking subtitles: updates every 1/fps frame
        srt_lines = []
        cue_idx = 1
        num_prog_steps = int(math.ceil(args.duration * fps))

        for step_idx in range(num_prog_steps):
            t_start = step_idx / fps
            t_end = min(args.duration, (step_idx + 1) / fps)
            if t_end <= t_start:
                continue

            frac = t_start / args.duration if args.duration > 0 else 1.0
            if mode == "linear":
                b = int(total_bytes * frac)
            else:
                b = int(total_bytes * (frac**2))

            pct = (b / total_bytes * 100.0) if total_bytes > 0 else 100.0
            sub_text = f"{fmt_size(b)} / {fmt_size(total_bytes)} ({pct:.1f}%)"

            srt_lines.append(f"{cue_idx}")
            srt_lines.append(f"{fmt_srt_time(t_start)} --> {fmt_srt_time(t_end)}")
            srt_lines.append(sub_text)
            srt_lines.append("")
            cue_idx += 1

        if args.hold > 0:
            t_start = args.duration
            t_end = args.duration + args.hold
            sub_text = f"{fmt_size(total_bytes)} / {fmt_size(total_bytes)} (100.0%)"
            srt_lines.append(f"{cue_idx}")
            srt_lines.append(f"{fmt_srt_time(t_start)} --> {fmt_srt_time(t_end)}")
            srt_lines.append(sub_text)
            srt_lines.append("")

        srt_file = output_dir / f"subtitles_{mode}.srt"
        with open(srt_file, "w") as f:
            f.write("\n".join(srt_lines) + "\n")

        # Output video
        video_clean = output_dir / f"progressive_{mode}.mp4"
        print(f"Creating clean {mode} progressive video...")
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_file.name,
                "-r",
                str(fps),
                "-vf",
                filter_base,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                video_clean.name,
            ],
            cwd=output_dir,
        )

        if args.annotate:
            video_annotated = output_dir / f"progressive_{mode}_annotated.mp4"
            print(f"Creating annotated {mode} progressive video...")
            subprocess.check_call(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    concat_file.name,
                    "-r",
                    str(fps),
                    "-vf",
                    f"{filter_base},subtitles={srt_file.name}:{sub_style}",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    video_annotated.name,
                ],
                cwd=output_dir,
            )

    print("Done! Output files in", output_dir)
    for mp4 in sorted(output_dir.glob("*.mp4")):
        print(" -", mp4)


if __name__ == "__main__":
    main()
