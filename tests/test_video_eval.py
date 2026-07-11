import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "tests" / "video_scenarios.json"
MODULE_PATH = ROOT_DIR / "tests" / "video_eval_runner.py"

SPEC = importlib.util.spec_from_file_location("video_eval_runner", MODULE_PATH)
video_eval_runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(video_eval_runner)


def load_scenarios():
    if not CONFIG_PATH.exists():
        pytest.skip(
            f"Missing scenario config: {CONFIG_PATH}. "
            "Create it from tests/video_scenarios.example.json."
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    base_dir = CONFIG_PATH.parent
    scenarios = []
    for scenario in config.get("scenarios", []):
        normalized = dict(scenario)
        normalized["video"] = video_eval_runner.normalize_video_path(base_dir, scenario["video"])

        annotated_output = scenario.get("annotated_output")
        if annotated_output:
            normalized["annotated_output"] = video_eval_runner.normalize_video_path(
                base_dir,
                annotated_output,
            )

        scenarios.append(normalized)

    return scenarios


SCENARIOS = load_scenarios()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[scenario["name"] for scenario in SCENARIOS])
def test_video_scenario_tracking(scenario):
    video_path = Path(scenario["video"])
    assert video_path.exists(), f"Video not found: {video_path}"

    report_dir = ROOT_DIR / "tests" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    annotated_output = scenario.get("annotated_output")
    if annotated_output:
        Path(annotated_output).parent.mkdir(parents=True, exist_ok=True)

    result = video_eval_runner.run_scenario(sys.executable, scenario, report_dir)

    failure_details = "\n".join(result["failures"])
    stderr = result["stderr"] or "<empty>"
    stdout = result["stdout"] or "<empty>"

    assert result["returncode"] == 0, (
        f"lock_and_track.py exited with code {result['returncode']}\n"
        f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
    )
    assert result["metrics"], f"No metrics produced for scenario {scenario['name']}"
    assert result["passed"], (
        f"Scenario {scenario['name']} failed thresholds.\n"
        f"Failures:\n{failure_details or '<none>'}\n\n"
        f"Metrics:\n{json.dumps(result['metrics'], indent=2)}"
    )


def test_static_threshold_example_passes_when_metrics_are_good():
    scenario = {
        "name": "static_youri",
        "thresholds": {
            "lock_acquired": True,
            "max_lock_frame": 15,
            "min_lock_ratio": 0.95,
            "max_times_lock_lost": 0,
            "require_ended_locked": True,
        },
    }
    metrics = {
        "lock_acquired": True,
        "lock_frame": 6,
        "lock_ratio": 0.99,
        "times_lock_lost": 0,
        "ended_locked": True,
    }

    passed, failures = video_eval_runner.evaluate_thresholds(scenario, metrics)

    assert passed is True
    assert failures == []


def test_crossing_threshold_example_fails_when_lock_is_lost_too_often():
    scenario = {
        "name": "crossing_people",
        "thresholds": {
            "lock_acquired": True,
            "min_lock_ratio": 0.80,
            "max_times_lock_lost": 1,
            "require_ended_locked": True,
        },
    }
    metrics = {
        "lock_acquired": True,
        "lock_ratio": 0.72,
        "times_lock_lost": 3,
        "ended_locked": False,
    }

    passed, failures = video_eval_runner.evaluate_thresholds(scenario, metrics)

    assert passed is False
    assert len(failures) == 3
