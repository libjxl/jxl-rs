#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Static cumulative stack usage analyzer for jxl-rs from jxl entry points down.

Analyzes cumulative stack usage across the call graph using rustc's -Z emit-stack-sizes,
parsing ELF .stack_sizes and disassembling functions via llvm-objdump.
"""

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

from elftools.elf.elffile import ELFFile

DEFAULT_TARGETS = [
    "x86_64-unknown-linux-gnu",
    "i686-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "armv7-unknown-linux-gnueabihf",
]

DEFAULT_BINARY = "jxl_cli"
DEFAULT_MAX_STACK = 16384  # 16 KiB
DEFAULT_TOP = 30


def decode_uleb128(data: bytes, offset: int) -> Tuple[int, int]:
    """Decode an unsigned LEB128 integer from data starting at offset."""
    result = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
    return result, offset


def find_demangler() -> Optional[str]:
    """Find the best available Rust/C++ symbol demangler."""
    for candidate in ["rustfilt", "llvm-cxxfilt"]:
        p = shutil.which(candidate)
        if p:
            return p
    for ver in range(22, 13, -1):
        p = shutil.which(f"llvm-cxxfilt-{ver}")
        if p:
            return p
    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        if os.path.isdir(path_dir):
            try:
                for entry in sorted(os.listdir(path_dir), reverse=True):
                    if entry.startswith("llvm-cxxfilt"):
                        full_path = os.path.join(path_dir, entry)
                        if os.access(full_path, os.X_OK):
                            return full_path
            except OSError:
                continue
    return shutil.which("c++filt")


def clean_short_name(dem: str) -> str:
    """Extract a clean, readable short function name suitable for Markdown tables."""
    # Strip crate disambiguator hashes (e.g. [273befe290477833])
    dem = re.sub(r"\[[0-9a-fA-F]{8,16}\]", "", dem)
    # Strip closure suffixes (e.g. ::{closure#0})
    dem = re.sub(r"::\{closure#\d+\}", "", dem)

    if ">::" in dem:
        left, method_part = dem.rsplit(">::", 1)
        if " as " in left:
            left = left.split(" as ")[-1]
        left = left.strip("<>")
        type_name = left.split("<")[0].split("::")[-1]
        method_name = method_part.split("<")[0].strip(":")
        clean = f"{type_name}::{method_name}" if type_name else method_name
    else:
        base = re.sub(r"::<[^>]*>", "", dem)
        base = base.split("<")[0].rstrip(":")
        parts = base.split("::")
        if len(parts) >= 2:
            clean = f"{parts[-2]}::{parts[-1]}"
        else:
            clean = parts[-1]

    # If the symbol remains mangled (starts with _R or _Z), truncate if excessively long
    if clean.startswith("_R") or clean.startswith("_Z"):
        if len(clean) > 28:
            clean = clean[:25] + "..."

    # Sanitize characters that break Markdown tables
    clean = clean.replace("|", "/").replace("`", "'")
    return clean


def demangle_symbols(symbols: List[str]) -> List[str]:
    """Batch-demangle Rust symbols using llvm-cxxfilt, rustfilt, or c++filt."""
    if not symbols:
        return []

    demangler = find_demangler()
    if not demangler:
        print(
            "Error: llvm-cxxfilt or c++filt is required to demangle symbols.",
            file=sys.stderr,
        )
        sys.exit(1)

    proc = subprocess.run(
        [demangler],
        input="\n".join(symbols),
        text=True,
        capture_output=True,
        check=True,
    )
    demangled = proc.stdout.splitlines()
    if len(demangled) != len(symbols):
        print(
            f"Error: Demangler '{demangler}' returned {len(demangled)} lines for {len(symbols)} input symbols.",
            file=sys.stderr,
        )
        sys.exit(1)

    return demangled


def is_panic_or_abort(demangled_name: str) -> bool:
    """Check if a function represents an abort, panic, unwind, or backtrace handler."""
    return any(
        pattern in demangled_name
        for pattern in (
            "core::panicking::",
            "std::panicking::",
            "std::sys::backtrace::",
            "std::backtrace_rs::",
            "panic_fmt",
            "slice_index_fail",
            "panic_bounds_check",
            "assert_failed",
            "unwrap_failed",
            "rust_begin_unwind",
            "handle_alloc_error",
            "rust_oom",
        )
    )


def is_jxl_symbol(demangled_name: str) -> bool:
    """Check if a function belongs to jxl, jxl_transforms, or jxl_simd."""
    return (
        demangled_name.startswith("jxl::")
        or demangled_name.startswith("<jxl::")
        or demangled_name.startswith("jxl_transforms::")
        or demangled_name.startswith("<jxl_transforms::")
        or demangled_name.startswith("jxl_simd::")
        or demangled_name.startswith("<jxl_simd::")
    )


def build_binary(
    target: str,
    target_dir: str,
    binary_name: str = DEFAULT_BINARY,
) -> Optional[str]:
    """Compile the final binary for target using -Z emit-stack-sizes."""
    env = os.environ.copy()
    env["RUSTFLAGS"] = (
        env.get("RUSTFLAGS", "") + " -Z emit-stack-sizes"
    ).strip()

    if target == "aarch64-unknown-linux-gnu":
        linker = env.get("CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER", "aarch64-linux-gnu-gcc")
        env["CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER"] = linker
        env["CC_aarch64_unknown_linux_gnu"] = linker
    elif target == "armv7-unknown-linux-gnueabihf":
        linker = env.get("CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_LINKER", "arm-linux-gnueabihf-gcc")
        env["CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_LINKER"] = linker
        env["CC_armv7_unknown_linux_gnueabihf"] = linker
    elif target == "i686-unknown-linux-gnu":
        linker = env.get("CARGO_TARGET_I686_UNKNOWN_LINUX_GNU_LINKER")
        if not linker and shutil.which("i686-linux-gnu-gcc"):
            linker = "i686-linux-gnu-gcc"
        if linker:
            env["CARGO_TARGET_I686_UNKNOWN_LINUX_GNU_LINKER"] = linker
            env["CC_i686_unknown_linux_gnu"] = linker

    cmd = [
        "cargo",
        "+nightly",
        "build",
        "--release",
        "--target",
        target,
        "--bin",
        binary_name,
        "--no-default-features",
        "--features",
        "all-simd",
        "--target-dir",
        target_dir,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(f"Error compiling {binary_name} for target {target}:", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return None

    binary_path = os.path.join(target_dir, target, "release", binary_name)
    if os.path.exists(binary_path):
        return binary_path

    print(f"Could not locate compiled binary for {binary_name} ({target})", file=sys.stderr)
    return None


def extract_stack_sizes_and_call_graph(
    binary_path: str,
) -> Tuple[Dict[str, int], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Extract per-function frame sizes, call graph, and tail-call graph from ELF binary."""
    name_to_size: Dict[str, int] = {}
    with open(binary_path, "rb") as f:
        elf = ELFFile(f)
        sec = elf.get_section_by_name(".stack_sizes")
        if not sec:
            return {}, {}, {}

        sec_data = sec.data()
        ptr_size = 4 if elf.elfclass == 32 else 8
        fmt = "<I" if ptr_size == 4 else "<Q"
        offset = 0
        entries: List[Tuple[int, int]] = []
        while offset < len(sec_data):
            addr = struct.unpack_from(fmt, sec_data, offset)[0]
            offset += ptr_size
            size, offset = decode_uleb128(sec_data, offset)
            entries.append((addr, size))

        symtab = elf.get_section_by_name(".symtab")
        addr_to_name: Dict[int, str] = {}
        if symtab:
            for sym in symtab.iter_symbols():
                if sym["st_value"] != 0 and sym["st_info"]["type"] == "STT_FUNC":
                    addr_to_name[sym["st_value"]] = sym.name

        for addr, sz in entries:
            name = addr_to_name.get(addr)
            if name:
                name_to_size[name] = max(name_to_size.get(name, 0), sz)

    objdump = shutil.which("llvm-objdump") or shutil.which("objdump")
    if not objdump:
        print("Error: llvm-objdump or objdump is required to extract call graph.", file=sys.stderr)
        sys.exit(1)

    proc = subprocess.Popen([objdump, "-d", binary_path], stdout=subprocess.PIPE, text=True)
    re_func = re.compile(r"^[0-9a-fA-F]+\s+<([^>]+)>:")
    re_call = re.compile(r"\b(?:call[ql]?|blx?)\s+[^<]*<([^>+]+)")
    re_tail = re.compile(r"\b(?:jmp[ql]?|b|b\.w)\s+[^<#*]*<([^>+]+)>")

    call_graph: Dict[str, Set[str]] = {}
    tail_graph: Dict[str, Set[str]] = {}
    current_func: Optional[str] = None

    for line in proc.stdout:
        m = re_func.match(line)
        if m:
            current_func = m.group(1)
            if current_func not in call_graph:
                call_graph[current_func] = set()
            if current_func not in tail_graph:
                tail_graph[current_func] = set()
            continue
        if current_func:
            mc = re_call.search(line)
            if mc:
                callee = mc.group(1)
                if callee != current_func:
                    call_graph[current_func].add(callee)
            mt = re_tail.search(line)
            if mt:
                callee = mt.group(1)
                if callee != current_func:
                    tail_graph[current_func].add(callee)
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"'{objdump} -d' failed with exit code {ret}")

    return name_to_size, call_graph, tail_graph


def solve_cumulative_stack(
    stack_sizes: Dict[str, int],
    call_graph: Dict[str, Set[str]],
    tail_graph: Dict[str, Set[str]],
    name_to_dem: Dict[str, str],
) -> Dict[str, Tuple[int, List[str]]]:
    """Compute maximum cumulative stack depth and call chain for each function via DFS."""
    memo: Dict[str, Tuple[int, List[str]]] = {}

    def dfs(node: str, visited: Set[str]) -> Tuple[int, List[str]]:
        if node in memo:
            return memo[node]
        dem = name_to_dem.get(node, node)
        if is_panic_or_abort(dem):
            memo[node] = (0, [node])
            return memo[node]
        if node in visited:
            return (stack_sizes.get(node, 0), [node])
        visited.add(node)
        self_sz = stack_sizes.get(node, 0)

        best_tot = self_sz
        best_path = [node]

        # Normal calls: callee executes while this function's stack frame is active
        for callee in call_graph.get(node, ()):
            c_tot, c_path = dfs(callee, visited)
            if self_sz + c_tot > best_tot:
                best_tot = self_sz + c_tot
                best_path = [node] + c_path

        # Tail calls: this function deallocates its frame before jumping,
        # so callee executes without this function's stack frame on stack
        for callee in tail_graph.get(node, ()):
            c_tot, c_path = dfs(callee, visited)
            if c_tot > best_tot:
                best_tot = c_tot
                best_path = [node] + c_path

        visited.remove(node)
        memo[node] = (best_tot, best_path)
        return memo[node]

    all_funcs = set(call_graph.keys()) | set(tail_graph.keys())
    for n in all_funcs:
        dfs(n, set())

    return memo


def get_jxl_entry_points(
    call_graph: Dict[str, Set[str]],
    tail_graph: Dict[str, Set[str]],
    name_to_dem: Dict[str, str],
) -> List[str]:
    """Identify entry points into jxl (functions in jxl with in-degree 0 from other jxl functions)."""
    all_funcs = set(call_graph.keys()) | set(tail_graph.keys())
    jxl_nodes = [n for n in all_funcs if is_jxl_symbol(name_to_dem.get(n, n))]
    jxl_node_set = set(jxl_nodes)

    # Count callers strictly within jxl (both normal and tail callers)
    callers_in_jxl: Dict[str, Set[str]] = {n: set() for n in jxl_nodes}
    for caller, callees in call_graph.items():
        if caller in jxl_node_set:
            for c in callees:
                if c in callers_in_jxl:
                    callers_in_jxl[c].add(caller)
    for caller, callees in tail_graph.items():
        if caller in jxl_node_set:
            for c in callees:
                if c in callers_in_jxl:
                    callers_in_jxl[c].add(caller)

    # An entry point is a jxl function not called by any other jxl function
    entry_points = [n for n in jxl_nodes if len(callers_in_jxl[n]) == 0]
    return entry_points


def main():
    parser = argparse.ArgumentParser(
        description="Check cumulative stack usage in jxl from entry points down."
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(DEFAULT_TARGETS),
        help=f"Comma-separated target triples (default: {','.join(DEFAULT_TARGETS)})",
    )
    parser.add_argument(
        "--binary",
        type=str,
        default=DEFAULT_BINARY,
        help=f"Binary target name to analyze (default: {DEFAULT_BINARY})",
    )
    parser.add_argument(
        "--max-stack",
        type=int,
        default=DEFAULT_MAX_STACK,
        help=f"Maximum allowed cumulative stack size in bytes (default: {DEFAULT_MAX_STACK})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help=f"Number of top functions to report per target (default: {DEFAULT_TOP})",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit summary in GitHub Markdown format to stdout",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="Path to write Markdown summary file",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="target",
        help="Cargo target directory",
    )

    args = parser.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    results_by_target = {}
    target_violations = []

    for target in targets:
        print(f"\n--> Building and analyzing `{args.binary}` for `{target}`...")
        bin_path = build_binary(
            target,
            args.target_dir,
            args.binary,
        )
        if bin_path is None:
            print(f"Failed to build `{args.binary}` on `{target}`", file=sys.stderr)
            sys.exit(1)

        stack_sizes, call_graph, tail_graph = extract_stack_sizes_and_call_graph(bin_path)
        all_funcs = set(call_graph.keys()) | set(tail_graph.keys())
        all_syms = list(all_funcs | set(stack_sizes.keys()))
        demangled_syms = demangle_symbols(all_syms)
        name_to_dem = dict(zip(all_syms, demangled_syms))

        cum_results = solve_cumulative_stack(stack_sizes, call_graph, tail_graph, name_to_dem)
        entry_points = get_jxl_entry_points(call_graph, tail_graph, name_to_dem)

        # Sort entry points by cumulative stack usage descending
        entry_points.sort(key=lambda k: cum_results[k][0], reverse=True)

        # Extract top 3 independent entry points / call chains
        top_3_entry_points = []
        seen_nodes: Set[str] = set()
        for ep in entry_points:
            if ep in seen_nodes:
                continue
            tot, path = cum_results[ep]
            top_3_entry_points.append((tot, path))
            seen_nodes.update(path)
            if len(top_3_entry_points) == 3:
                break

        # Also get all jxl functions sorted by cumulative stack
        jxl_functions = [n for n in all_funcs if is_jxl_symbol(name_to_dem.get(n, n))]
        jxl_functions.sort(key=lambda k: cum_results[k][0], reverse=True)

        results_by_target[target] = {
            "stack_sizes": stack_sizes,
            "call_graph": call_graph,
            "name_to_dem": name_to_dem,
            "cum_results": cum_results,
            "entry_points": entry_points,
            "top_3": top_3_entry_points,
            "jxl_functions": jxl_functions,
        }

    # Console reporting
    for target, data in results_by_target.items():
        cum_results = data["cum_results"]
        stack_sizes = data["stack_sizes"]
        name_to_dem = data["name_to_dem"]
        top_3 = data["top_3"]
        jxl_functions = data["jxl_functions"]

        if not top_3:
            print(f"\nNo jxl entry points analyzed for {target}.")
            continue

        highest_cum = top_3[0][0]
        highest_root = top_3[0][1][0]
        highest_dem = name_to_dem.get(highest_root, highest_root)

        print(f"\nTarget: {target}")
        print("-" * 80)
        print(f"  jxl functions analyzed: {len(jxl_functions):,}")
        print(f"  Top Cumulative Stack in jxl: {highest_cum:,} Bytes ({highest_cum / 1024:.2f} KiB) in `{highest_dem}`")

        if highest_cum >= args.max_stack:
            target_violations.append((target, highest_cum, highest_dem))

        print(f"\n  Top 3 Cumulative Stacks from jxl Entry Points Down:")
        for idx, (tot, path) in enumerate(top_3, 1):
            root = path[0]
            root_dem = name_to_dem.get(root, root)
            root_self = stack_sizes.get(root, 0)
            print(f"\n    #{idx}: {tot:6d} B ({tot / 1024:5.2f} KiB) [frame: {root_self:5d} B]: {root_dem}")
            print("        Call chain:")
            for p in path:
                p_self = stack_sizes.get(p, 0)
                p_dem = name_to_dem.get(p, p)
                print(f"          -> {p_self:5d} B : {p_dem}")

        print(f"\n  Top {min(args.top, len(jxl_functions))} jxl Functions by Cumulative Stack:")
        for idx, sym in enumerate(jxl_functions[: args.top], 1):
            tot, _ = cum_results[sym]
            self_sz = stack_sizes.get(sym, 0)
            dem = name_to_dem.get(sym, sym)
            print(f"    {idx:2d}. {tot:6d} B ({tot / 1024:5.2f} KiB) [frame: {self_sz:5d} B]: {dem}")

    # Markdown Summary
    if args.summary_file or args.markdown:
        md = []
        md.append("## Cumulative Stack Usage Summary (jxl Entry Points Down)\n")
        md.append(f"Threshold: **{args.max_stack:,} Bytes ({args.max_stack // 1024} KiB)**\n")
        md.append("| Target Architecture | #1 Cumulative | #2 Cumulative | #3 Cumulative | Compliant |")
        md.append("|---|---:|---:|---:|:---:|")
        for target, data in results_by_target.items():
            top_3 = data["top_3"]
            if not top_3:
                continue
            h_cum = top_3[0][0]
            status = "❌ Violation" if h_cum >= args.max_stack else "✅ OK"
            stacks_str = []
            for i in range(3):
                if i < len(top_3):
                    tot, path = top_3[i]
                    dem = data["name_to_dem"].get(path[0], path[0])
                    fn_name = clean_short_name(dem)
                    stacks_str.append(f"**{tot:,} B** (`{fn_name}`)")
                else:
                    stacks_str.append("-")
            md.append(f"| `{target}` | {stacks_str[0]} | {stacks_str[1]} | {stacks_str[2]} | {status} |")

        md.append("\n### Top 3 Call Chains per Target\n")
        for target, data in results_by_target.items():
            top_3 = data["top_3"]
            stack_sizes = data["stack_sizes"]
            name_to_dem = data["name_to_dem"]
            md.append(f"<details><summary><b>{target}</b> (Top 3 Call Chains)</summary>\n")
            for idx, (tot, path) in enumerate(top_3, 1):
                root = path[0]
                dem = name_to_dem.get(root, root).replace("`", "'").replace("|", "\\|")
                md.append(f"#### #{idx}: {tot:,} B ({tot / 1024:.2f} KiB) - `{dem}`\n")
                md.append("```text")
                for p in path:
                    p_self = stack_sizes.get(p, 0)
                    p_dem = name_to_dem.get(p, p)
                    md.append(f"  -> {p_self:5d} B : {p_dem}")
                md.append("```\n")
            md.append("</details>\n")

        if target_violations:
            md.append(f"❌ **FAILURE**: {len(target_violations)} target(s) exceed {args.max_stack:,} B ({args.max_stack // 1024} KiB) cumulative stack budget:\n")
            for t, sz, fn in target_violations:
                dem = fn.replace("`", "'")
                md.append(f"- `{t}`: **{sz:,} B** in `{dem}`")
        else:
            md.append(f"✅ **SUCCESS**: All targets within {args.max_stack:,} B ({args.max_stack // 1024} KiB) cumulative stack budget!\n")

        summary_content = "\n".join(md)
        if args.summary_file:
            with open(args.summary_file, "a") as f:
                f.write("\n" + summary_content + "\n")
        if args.markdown:
            print("\n" + summary_content)

    print()
    if target_violations:
        print(f"FAILURE: {len(target_violations)} target(s) exceed {args.max_stack} B ({args.max_stack // 1024} KiB) cumulative stack budget:")
        for t, sz, fn in target_violations:
            print(f"  - {t}: {sz:,} B in {fn}")
        sys.exit(1)
    else:
        print(f"SUCCESS: All targets within {args.max_stack} B ({args.max_stack // 1024} KiB) cumulative stack budget!")


if __name__ == "__main__":
    main()
