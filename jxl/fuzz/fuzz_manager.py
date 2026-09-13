#!/usr/bin/env python3
"""
Fuzz Manager for jxl-rs.
Orchestrates multi-target fuzzing campaigns across available CPU cores,
monitors real-time progress, stops runs cleanly, and triages discovered crashes.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_TARGET_CORES = {
    "decode_diff": 16,
    "decode": 12,
    "decode_progressive": 12,
    "decode_progressive_parallel": 12,
    "decode_parallel": 8,
    "decode_header": 4,
}

FUZZ_DIR = Path(__file__).resolve().parent
PROJECT_DIR = FUZZ_DIR.parent
TARGET_BIN_DIR = FUZZ_DIR / "target" / "x86_64-unknown-linux-gnu" / "release"
CORPUS_DIR = FUZZ_DIR / "corpus"
DICT_FILE = FUZZ_DIR / "jxl.dict"
RUNS_DIR = FUZZ_DIR / "runs"
ARTIFACTS_DIR = FUZZ_DIR / "artifacts"
PID_FILE = RUNS_DIR / "fuzz.pids"
STATE_FILE = RUNS_DIR / "state.json"


def parse_time_duration(duration_str: str) -> int:
    """Parses duration string like '8h', '30m', '3600s', '3600' into seconds."""
    if not duration_str:
        return 0
    duration_str = duration_str.strip().lower()
    if duration_str.endswith("h"):
        return int(float(duration_str[:-1]) * 3600)
    elif duration_str.endswith("m"):
        return int(float(duration_str[:-1]) * 60)
    elif duration_str.endswith("s"):
        return int(float(duration_str[:-1]))
    return int(duration_str)


def build_targets():
    """Ensures all fuzz targets are built."""
    print("==> Building fuzz targets with AddressSanitizer and optimizations...")
    cmd = ["cargo", "+nightly", "fuzz", "build"]
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)


def get_active_pids() -> dict:
    """Reads PID file and returns dict of target -> list of PIDs that are currently running."""
    if not PID_FILE.exists():
        return {}
    try:
        with open(PID_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        return {}

    active = {}
    for target, pids in data.items():
        alive_pids = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive_pids.append(pid)
            except OSError:
                pass
        if alive_pids:
            active[target] = alive_pids
    return active


def save_pids(pid_dict: dict):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PID_FILE, "w") as f:
        json.dump(pid_dict, f, indent=2)


def allocate_cores(total_cores: int, target_weights: dict) -> dict:
    """Proportionally distributes total_cores among targets."""
    base_sum = sum(target_weights.values())
    allocation = {}
    allocated_sum = 0
    for target, weight in target_weights.items():
        count = max(1, round(total_cores * (weight / base_sum)))
        allocation[target] = count
        allocated_sum += count

    # Adjust difference
    diff = total_cores - allocated_sum
    if diff != 0:
        sorted_targets = sorted(target_weights.keys(), key=lambda t: target_weights[t], reverse=True)
        idx = 0
        while diff > 0:
            allocation[sorted_targets[idx % len(sorted_targets)]] += 1
            diff -= 1
            idx += 1
        while diff < 0:
            for t in reversed(sorted_targets):
                if allocation[t] > 1 and diff < 0:
                    allocation[t] -= 1
                    diff += 1
    return allocation


def start_fuzzing(args):
    active = get_active_pids()
    if active:
        print(f"Error: Fuzzers already running! (PIDs: {active})")
        print("Run 'fuzz_manager.py stop' or 'fuzz_manager.py status' first.")
        sys.exit(1)

    if not args.no_build:
        build_targets()

    # Determine targets and core allocation
    if args.targets:
        selected_targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    else:
        selected_targets = list(DEFAULT_TARGET_CORES.keys())

    total_cores = args.cores if args.cores else (os.cpu_count() or 4)
    target_weights = {t: DEFAULT_TARGET_CORES.get(t, 4) for t in selected_targets}
    cores_per_target = allocate_cores(total_cores, target_weights)

    duration_secs = parse_time_duration(args.duration) if args.duration else 0
    timeout_secs = args.timeout or 25
    rss_limit_mb = args.rss_limit or 3072

    print(f"\n=======================================================")
    print(f"  Starting Overnight Fuzzing on {total_cores} Cores")
    if duration_secs > 0:
        print(f"  Duration: {args.duration} ({duration_secs}s)")
    else:
        print(f"  Duration: Indefinite (until stopped)")
    print(f"  Per-input timeout: {timeout_secs}s | RSS limit: {rss_limit_mb}MB")
    print(f"=======================================================\n")

    pids_by_target = {}
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for target in selected_targets:
        workers = cores_per_target[target]
        bin_path = TARGET_BIN_DIR / target
        if not bin_path.exists():
            print(f"Error: Binary {bin_path} not found. Run with cargo +nightly fuzz build first.")
            sys.exit(1)

        target_corpus = CORPUS_DIR / target
        target_corpus.mkdir(parents=True, exist_ok=True)
        target_artifacts = ARTIFACTS_DIR / target
        target_artifacts.mkdir(parents=True, exist_ok=True)
        target_run_dir = RUNS_DIR / target
        target_run_dir.mkdir(parents=True, exist_ok=True)

        stop_file = target_run_dir / "STOP"
        if stop_file.exists():
            stop_file.unlink()

        cmd = [
            str(bin_path),
            str(target_corpus),
            f"-dict={DICT_FILE}",
            f"-artifact_prefix={target_artifacts}/",
            f"-jobs=1000000",
            f"-workers={workers}",
            f"-timeout={timeout_secs}",
            f"-rss_limit_mb={rss_limit_mb}",
            "-ignore_crashes=1",
            f"-stop_file={stop_file}",
        ]
        if duration_secs > 0:
            cmd.append(f"-max_total_time={duration_secs}")

        master_log_path = target_run_dir / "master.log"
        master_log = open(master_log_path, "w")

        print(f"  [{target:28s}] Launching {workers:2d} workers in {target_run_dir.relative_to(PROJECT_DIR)}...")
        proc = subprocess.Popen(
            cmd,
            cwd=target_run_dir,
            stdout=master_log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setpgrp,
        )
        pids_by_target[target] = [proc.pid]

    save_pids(pids_by_target)
    with open(STATE_FILE, "w") as f:
        json.dump(
            {
                "start_time": time.time(),
                "duration_secs": duration_secs,
                "cores": total_cores,
                "allocation": cores_per_target,
            },
            f,
            indent=2,
        )

    print(f"\n==> All targets launched successfully!")
    print(f"  View status:  python3 jxl/fuzz/fuzz_manager.py status")
    print(f"  Stop fuzzers: python3 jxl/fuzz/fuzz_manager.py stop")
    print(f"  Triage bugs:  python3 jxl/fuzz/fuzz_manager.py triage\n")


def stop_fuzzing(args):
    active = get_active_pids()
    if not active:
        print("No active fuzzers found.")
        return

    print("==> Stopping fuzzers cleanly...")
    # 1. Touch stop files so libFuzzer exits gracefully
    for target in active.keys():
        stop_file = RUNS_DIR / target / "STOP"
        try:
            stop_file.touch()
        except Exception:
            pass

    time.sleep(1)

    # 2. Send SIGINT / SIGTERM to master processes
    for target, pids in active.items():
        for pid in pids:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass

    # Wait up to 5 seconds for processes to terminate
    deadline = time.time() + 5
    while time.time() < deadline:
        active = get_active_pids()
        if not active:
            break
        time.sleep(0.5)

    # Force kill if any remaining
    if active:
        print("Force killing remaining processes...")
        for target, pids in active.items():
            for pid in pids:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except Exception:
                    pass

    if PID_FILE.exists():
        PID_FILE.unlink()

    print("==> All fuzzers stopped.")


def parse_worker_stats(run_dir: Path) -> dict:
    """Parses latest cov, ft, corp, exec/s, and total runs from worker log files."""
    stats = {
        "execs": 0,
        "cov": 0,
        "ft": 0,
        "corp": 0,
        "corp_kb": 0,
        "exec_s": 0,
        "max_rss": 0,
        "active_workers": 0,
    }

    log_pattern = re.compile(
        r"#(\d+)\s+(?:pulse|NEW|REDUCE|RELOAD|DONE|INITED)\s+cov:\s+(\d+)\s+ft:\s+(\d+)\s+corp:\s+(\d+)(?:/(\d+)Kb)?.*?exec/s:\s+(\d+)\s+rss:\s+(\d+)Mb"
    )

    log_files = list(run_dir.glob("fuzz-*.log"))
    stats["total_workers"] = len(log_files)

    for log_file in log_files:
        try:
            # Read last 8KB of log
            with open(log_file, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 8192))
                lines = f.read().decode("utf-8", errors="ignore").splitlines()

            last_match = None
            for line in reversed(lines):
                m = log_pattern.search(line)
                if m:
                    last_match = m
                    break

            if last_match:
                runs = int(last_match.group(1))
                cov = int(last_match.group(2))
                ft = int(last_match.group(3))
                corp = int(last_match.group(4))
                corp_kb = int(last_match.group(5) or 0)
                exec_s = int(last_match.group(6))
                rss = int(last_match.group(7))

                stats["execs"] += runs
                stats["cov"] = max(stats["cov"], cov)
                stats["ft"] = max(stats["ft"], ft)
                stats["corp"] = max(stats["corp"], corp)
                stats["corp_kb"] = max(stats["corp_kb"], corp_kb)
                stats["exec_s"] += exec_s
                stats["max_rss"] = max(stats["max_rss"], rss)
        except Exception:
            pass

    return stats


def show_status(args):
    active = get_active_pids()
    state = {}
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
        except Exception:
            pass

    start_time = state.get("start_time", 0)
    uptime_str = "Not running"
    if start_time and active:
        elapsed = int(time.time() - start_time)
        hrs = elapsed // 3600
        mins = (elapsed % 3600) // 60
        secs = elapsed % 60
        uptime_str = f"{hrs:02d}h {mins:02d}m {secs:02d}s"

    print("\n" + "=" * 90)
    print(f" JXL-RS FUZZING STATUS   |   Status: {'RUNNING' if active else 'IDLE'}   |   Uptime: {uptime_str}")
    print("=" * 90)

    header = f"{'Target':<28} {'Workers':<9} {'Exec/s':<9} {'Coverage':<12} {'Features':<10} {'Corpus':<10} {'Crashes':<8}"
    print(header)
    print("-" * 90)

    total_exec_s = 0
    total_workers = 0
    total_crashes = 0
    all_targets = sorted(list(set(list(DEFAULT_TARGET_CORES.keys()) + list(active.keys()))))

    for target in all_targets:
        run_dir = RUNS_DIR / target
        stats = parse_worker_stats(run_dir) if run_dir.exists() else {}
        is_active = target in active
        workers_str = f"{len(active.get(target, [])) if is_active else 0} active"
        exec_s = stats.get("exec_s", 0) if is_active else 0
        total_exec_s += exec_s
        total_workers += len(active.get(target, [])) if is_active else 0

        cov_str = str(stats.get("cov", "-"))
        ft_str = str(stats.get("ft", "-"))

        target_corpus = CORPUS_DIR / target
        corpus_count = len(list(target_corpus.glob("*"))) if target_corpus.exists() else 0

        target_artifacts = ARTIFACTS_DIR / target
        crashes = list(target_artifacts.glob("crash-*")) if target_artifacts.exists() else []
        timeouts = list(target_artifacts.glob("timeout-*")) if target_artifacts.exists() else []
        ooms = list(target_artifacts.glob("oom-*")) if target_artifacts.exists() else []
        crash_count = len(crashes) + len(timeouts) + len(ooms)
        total_crashes += crash_count

        crash_display = f"{crash_count}" if crash_count > 0 else "0"
        if crash_count > 0:
            crash_display = f"\033[91m{crash_display} (!)\033[0m"

        target_display = f"\033[92m{target}\033[0m" if is_active else target
        w = 37 if is_active else 28
        print(
            f"{target_display:<{w}} {workers_str:<9} {exec_s:<9} {cov_str:<12} {ft_str:<10} {corpus_count:<10} {crash_display}"
        )

    print("-" * 90)
    print(
        f"{'TOTAL':<28} {total_workers:<9} {total_exec_s:<9} {'-':<12} {'-':<10} {'-':<10} {total_crashes}"
    )
    print("=" * 90 + "\n")

    if total_crashes > 0:
        print(f"\033[91mFound {total_crashes} artifacts! Run 'python3 jxl/fuzz/fuzz_manager.py triage' to inspect.\033[0m\n")


def triage_artifacts(args):
    """Scans ARTIFACTS_DIR for any crash/timeout/oom and runs reproducer."""
    print("\n" + "=" * 80)
    print(" ARTIFACT TRIAGE & CRASH INSPECTION")
    print("=" * 80 + "\n")

    found_artifacts = []
    for target_dir in ARTIFACTS_DIR.glob("*"):
        if not target_dir.is_dir():
            continue
        target_name = target_dir.name
        for art in target_dir.glob("*"):
            if art.is_file():
                found_artifacts.append((target_name, art))

    # Prioritize crashes and OOMs before timeouts
    def artifact_priority(item):
        name = item[1].name
        if name.startswith("crash"):
            return 0
        elif name.startswith("oom"):
            return 1
        elif name.startswith("leak"):
            return 2
        return 3

    found_artifacts.sort(key=artifact_priority)

    if getattr(args, "only_crashes", False):
        found_artifacts = [
            (t, a) for (t, a) in found_artifacts if not a.name.startswith("timeout") and not a.name.startswith("slow")
        ]

    if not found_artifacts:
        print("No matching artifacts found in artifacts/ directory.")
        return

    timeout_sec = getattr(args, "timeout", None) or 5
    print(f"Found {len(found_artifacts)} artifact(s) (reproducer timeout: {timeout_sec}s):\n")
    for target, art in found_artifacts:
        print(f"--------------------------------------------------------------------------------")
        print(f"Target:   {target}")
        print(f"Artifact: {art.relative_to(PROJECT_DIR)} ({art.stat().st_size} bytes)")
        print(f"--------------------------------------------------------------------------------")

        bin_path = TARGET_BIN_DIR / target
        if not bin_path.exists():
            print(f"  Warning: Target binary {bin_path} not found. Build target first.")
            continue

        cmd = [str(bin_path), str(art)]
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
            )
            lines = res.stdout.strip().splitlines()
            output_snippet = "\n".join(lines[-30:]) if len(lines) > 30 else "\n".join(lines)
            print(output_snippet)
        except subprocess.TimeoutExpired:
            print(f"  \033[93m[TIMEOUT] Reproduction timed out after {timeout_sec}s (infinite loop or pathological input)\033[0m")
        except Exception as e:
            print(f"  Error running reproducer: {e}")
        print("\n")


def main():
    parser = argparse.ArgumentParser(description="JXL-RS Fuzzing Campaign Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # start
    p_start = subparsers.add_parser("start", help="Start overnight fuzzing campaign")
    p_start.add_argument(
        "--duration",
        "-d",
        type=str,
        default="8h",
        help="Campaign duration, e.g. 8h, 12h, 30m, or 0 for indefinite (default: 8h)",
    )
    p_start.add_argument(
        "--cores",
        "-c",
        type=int,
        default=None,
        help="Total CPU cores to utilize (default: all available cores)",
    )
    p_start.add_argument(
        "--targets",
        "-t",
        type=str,
        default=None,
        help="Comma-separated list of targets to fuzz (default: all)",
    )
    p_start.add_argument(
        "--timeout",
        type=int,
        default=25,
        help="Per-input execution timeout in seconds (default: 25)",
    )
    p_start.add_argument(
        "--rss-limit",
        type=int,
        default=3072,
        help="Per-worker RSS limit in MB (default: 3072)",
    )
    p_start.add_argument(
        "--no-build",
        action="store_true",
        help="Skip rebuilding fuzz targets before starting",
    )

    # stop
    p_stop = subparsers.add_parser("stop", help="Cleanly stop all running fuzzers")

    # status
    p_status = subparsers.add_parser("status", help="Show real-time fuzzing status")

    # triage
    p_triage = subparsers.add_parser("triage", help="Triage discovered crashes and artifacts")
    p_triage.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Per-artifact reproducer timeout in seconds (default: 5)",
    )
    p_triage.add_argument(
        "--only-crashes",
        action="store_true",
        help="Only triage crashes and OOMs, skipping timeouts",
    )

    args = parser.parse_args()

    if args.command == "start":
        start_fuzzing(args)
    elif args.command == "stop":
        stop_fuzzing(args)
    elif args.command == "status":
        show_status(args)
    elif args.command == "triage":
        triage_artifacts(args)


if __name__ == "__main__":
    main()
