import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCK_AND_TRACK_PATH = ROOT_DIR / "src" / "lock_and_track.py"
DEFAULT_REPORT_DIR = ROOT_DIR / "tests" / "reports"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run video evaluation scenarios for lock_and_track.py."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the JSON scenario config file.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch lock_and_track.py.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory where per-scenario JSON reports are stored.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_report_dir(report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)


def normalize_video_path(base_dir, video_path):
    path = Path(video_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def sanitize_name(name):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def build_command(python_executable, scenario, metrics_output_path):
    command = [
        python_executable,
        str(LOCK_AND_TRACK_PATH),
        "--video",
        str(scenario["video"]),
        "--headless",
        "--metrics-output",
        str(metrics_output_path),
    ]

    output_path = scenario.get("annotated_output")
    if output_path:
        command.extend(["--output", str(output_path)])

    max_frames = scenario.get("max_frames")
    if max_frames is not None:
        command.extend(["--max-frames", str(max_frames)])

    return command


def evaluate_thresholds(scenario, metrics):
    thresholds = scenario.get("thresholds", {})
    failures = []

    if "lock_acquired" in thresholds:
        expected = thresholds["lock_acquired"]
        if bool(metrics.get("lock_acquired")) != bool(expected):
            failures.append(
                f"lock_acquired expected {expected}, got {metrics.get('lock_acquired')}"
            )

    if "max_lock_frame" in thresholds:
        lock_frame = metrics.get("lock_frame")
        if lock_frame is None or lock_frame > thresholds["max_lock_frame"]:
            failures.append(
                f"lock_frame should be <= {thresholds['max_lock_frame']}, got {lock_frame}"
            )

    if "min_lock_ratio" in thresholds:
        lock_ratio = float(metrics.get("lock_ratio", 0.0))
        if lock_ratio < thresholds["min_lock_ratio"]:
            failures.append(
                f"lock_ratio should be >= {thresholds['min_lock_ratio']}, got {lock_ratio:.3f}"
            )

    if "min_target_visibility_ratio" in thresholds:
        visibility_ratio = float(metrics.get("target_visibility_ratio", 0.0))
        if visibility_ratio < thresholds["min_target_visibility_ratio"]:
            failures.append(
                "target_visibility_ratio should be >= "
                f"{thresholds['min_target_visibility_ratio']}, got {visibility_ratio:.3f}"
            )

    if "max_times_lock_lost" in thresholds:
        times_lock_lost = int(metrics.get("times_lock_lost", 0))
        if times_lock_lost > thresholds["max_times_lock_lost"]:
            failures.append(
                f"times_lock_lost should be <= {thresholds['max_times_lock_lost']}, got {times_lock_lost}"
            )

    if "max_consecutive_lost_frames" in thresholds:
        max_lost = int(metrics.get("max_consecutive_lost_frames", 0))
        if max_lost > thresholds["max_consecutive_lost_frames"]:
            failures.append(
                "max_consecutive_lost_frames should be <= "
                f"{thresholds['max_consecutive_lost_frames']}, got {max_lost}"
            )

    if thresholds.get("require_ended_locked") and not metrics.get("ended_locked"):
        failures.append("tracking should still be locked on the final frame")

    if thresholds.get("require_target_present_at_end") and not metrics.get("target_present_at_end"):
        failures.append("target should be visible on the final processed frame")

    return len(failures) == 0, failures


def run_scenario(python_executable, scenario, report_dir):
    scenario_name = scenario["name"]
    metrics_output_path = report_dir / f"{sanitize_name(scenario_name)}_metrics.json"
    annotated_output = scenario.get("annotated_output")
    if annotated_output:
        Path(annotated_output).parent.mkdir(parents=True, exist_ok=True)
    command = build_command(python_executable, scenario, metrics_output_path)

    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    )

    metrics = {}
    if metrics_output_path.exists():
        with open(metrics_output_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    passed = completed.returncode == 0
    failures = []

    if not passed and not metrics:
        failures.append("lock_and_track.py did not produce a metrics report")

    if metrics:
        threshold_passed, threshold_failures = evaluate_thresholds(scenario, metrics)
        passed = passed and threshold_passed
        failures.extend(threshold_failures)

    return {
        "name": scenario_name,
        "video": str(scenario["video"]),
        "passed": passed and len(failures) == 0,
        "returncode": completed.returncode,
        "metrics_path": str(metrics_output_path),
        "metrics": metrics,
        "failures": failures,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def print_summary(results):
    print("\n=== Video Evaluation Summary ===")
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['name']} -> {result['video']}")
        if result["metrics"]:
            metrics = result["metrics"]
            print(
                "  "
                f"lock_frame={metrics.get('lock_frame')} | "
                f"lock_ratio={metrics.get('lock_ratio', 0.0):.3f} | "
                f"times_lock_lost={metrics.get('times_lock_lost')} | "
                f"ended_locked={metrics.get('ended_locked')}"
            )
        for failure in result["failures"]:
            print(f"  - {failure}")


def main():
    args = parse_args()
    config = load_config(args.config)
    base_dir = args.config.resolve().parent
    scenarios = config.get("scenarios", [])

    ensure_report_dir(args.report_dir)

    normalized_scenarios = []
    for scenario in scenarios:
        normalized = dict(scenario)
        normalized["video"] = normalize_video_path(base_dir, scenario["video"])

        annotated_output = scenario.get("annotated_output")
        if annotated_output:
            normalized["annotated_output"] = normalize_video_path(base_dir, annotated_output)

        normalized_scenarios.append(normalized)

    results = []
    for scenario in normalized_scenarios:
        results.append(run_scenario(args.python, scenario, args.report_dir))

    print_summary(results)

    summary_path = args.report_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    if any(not result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
