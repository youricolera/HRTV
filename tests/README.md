# Video evaluation for `lock_and_track.py`

This test setup lets you evaluate the tracker on a list of videos and validate
simple success criteria for each scenario.

## What it measures

`src/lock_and_track.py` now writes a JSON report with:

- `lock_acquired`: whether Youri was ever locked
- `lock_frame`: first frame where the lock was acquired
- `lock_ratio`: fraction of processed frames where the tracker stayed locked
- `target_visibility_ratio`: fraction of frames where the target track was found
- `times_lock_lost`: how many times the tracker fully unlocked
- `max_consecutive_lost_frames`: longest temporary loss streak
- `ended_locked`: whether the tracker was still locked on the last frame
- `target_present_at_end`: whether the target was visible on the last processed frame

## Prepare your scenarios

1. Copy `tests/video_scenarios.example.json` to `tests/video_scenarios.json`.
2. Replace the example video paths with your real files.
3. Adjust thresholds scenario by scenario.

## Run the evaluation suite with pytest

```powershell
pytest tests\test_video_eval.py -s
```

To run only one scenario:

```powershell
pytest tests\test_video_eval.py -s -k static_youri_front_webcam
```

The helper runner can still be launched directly if needed:

```powershell
python tests\video_eval_runner.py --config tests\video_scenarios.json
```

## First scenario to create

For the first baseline video, use one video where:

- Youri is alone
- Youri faces the webcam
- movement is minimal
- lighting is stable

Recommended success criteria for this baseline:

- fast first lock
- no full unlock
- lock maintained until the end

## Then add harder videos

Good next scenarios:

- another person crosses in front of Youri
- two people remain in frame for a long time
- partial occlusion
- Youri moves left/right and turns slightly
- distance to the camera changes

Pytest uses the same runner under the hood and writes individual metrics files
to `tests/reports/`. The direct runner also writes `tests/reports/summary.json`.
