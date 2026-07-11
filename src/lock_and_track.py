import argparse
import cv2
import json
import pickle
import face_recognition
import numpy as np
from ultralytics import YOLO


# -----------------------------
# Config
# -----------------------------
YOLO_MODEL_PATH = "models/yolov8n.pt"
ENCODINGS_PATH = "youri_encodings_enriched.pkl"

CAMERA_INDEX = 0
FRAME_WIDTH = 3840 
FRAME_HEIGHT = 2160 
DISPLAY_MAX_WIDTH = 1280
DISPLAY_MAX_HEIGHT = 720

PERSON_CLASS_ID = 0
FACE_TOLERANCE = 0.55
RECOGNITION_INTERVAL = 1  # Run face recognition every N frames when unlocked
REASSOCIATE_MAX_DISTANCE = 140
REASSOCIATE_MIN_SCORE = 0.45
REASSOCIATE_MIN_MARGIN = 0.12
REASSOCIATE_HOLD_FRAMES = 3
MAX_REASSOCIATE_FRAMES = 8
TRACK_VALIDATION_MIN_SCORE = 0.40
TRACK_VALIDATION_MIN_APPEARANCE = 0.30
TARGET_HIST_ALPHA = 0.2
OCCLUSION_IOU_THRESHOLD = 0.15
POST_OCCLUSION_REASSOCIATE_FRAMES = 8


# -----------------------------
# Helpers
# -----------------------------
def load_known_encodings(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lock on a known face, then track that person in a webcam or video."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a video file. If omitted, the webcam is used.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the annotated output video.",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        help="Optional path to save a JSON evaluation report.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the OpenCV window. Useful for automated video evaluation.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on the number of processed frames.",
    )
    parser.add_argument(
        "--display-max-width",
        type=int,
        default=DISPLAY_MAX_WIDTH,
        help="Maximum display width for the preview window.",
    )
    parser.add_argument(
        "--display-max-height",
        type=int,
        default=DISPLAY_MAX_HEIGHT,
        help="Maximum display height for the preview window.",
    )
    return parser.parse_args()


def build_metrics(video_path, device):
    return {
        "video_path": video_path,
        "device": device,
        "frames_processed": 0,
        "frames_with_people": 0,
        "frames_locked": 0,
        "frames_target_visible": 0,
        "lock_acquired": False,
        "lock_frame": None,
        "initial_target_track_id": None,
        "final_target_track_id": None,
        "times_lock_acquired": 0,
        "times_lock_lost": 0,
        "max_consecutive_lost_frames": 0,
        "occlusion_frames": 0,
        "reassociation_successes": 0,
        "target_present_at_end": False,
        "ended_locked": False,
        "lock_ratio": 0.0,
        "target_visibility_ratio": 0.0,
    }


def finalize_metrics(metrics):
    frames_processed = metrics["frames_processed"]
    if frames_processed > 0:
        metrics["lock_ratio"] = metrics["frames_locked"] / frames_processed
        metrics["target_visibility_ratio"] = (
            metrics["frames_target_visible"] / frames_processed
        )
    return metrics


def save_metrics(metrics, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def resize_for_display(frame, max_width, max_height):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)

    if scale >= 1.0:
        return frame

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def bbox_iou(box_a, box_b):
    """
    Compute IoU between two boxes in xyxy format.
    box = (x1, y1, x2, y2)
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0

    return inter_area / union


def face_box_to_xyxy(face_location):
    """
    face_recognition returns (top, right, bottom, left)
    Convert to (x1, y1, x2, y2)
    """
    top, right, bottom, left = face_location
    return (left, top, right, bottom)


def draw_label(frame, text, org, color):
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        4
    )
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def center_distance(box_a, box_b):
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    return float(np.hypot(ax - bx, ay - by))


def extract_target_patch(frame, bbox):
    """
    Extract the upper-body region from a person's bbox.

    The goal is to keep a visually stable area for re-identification:
    the torso and shoulders are usually more reliable than the full body
    when people cross or partially occlude each other.

    Returns a cropped image patch, or None if the bbox is invalid or too small.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    box_h = y2 - y1
    upper_body_y2 = y1 + max(1, int(box_h * 0.6))
    patch = frame[y1:upper_body_y2, x1:x2]

    if patch.size == 0 or patch.shape[0] < 10 or patch.shape[1] < 10:
        return None

    return patch


def compute_appearance_histogram(frame, bbox):
    """
    Build a simple appearance signature for one person.

    The signature is a normalized 2D HSV histogram computed on the upper-body
    patch. This gives a compact representation of clothing colors that can be
    compared later if the tracker changes the target's track_id.

    Returns the histogram, or None if no valid patch can be extracted.
    """
    patch = extract_target_patch(frame, bbox)
    if patch is None:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def update_target_histogram(current_hist, new_hist, alpha=TARGET_HIST_ALPHA):
    """
    Update the stored target appearance with exponential smoothing.

    Instead of replacing the target histogram abruptly, we blend the previous
    appearance with the new observation. This makes the model less sensitive to
    lighting changes or small bbox variations while keeping the target identity
    stable over time.
    """
    if new_hist is None:
        return current_hist
    if current_hist is None:
        return new_hist

    blended = (1.0 - alpha) * current_hist + alpha * new_hist
    norm = np.linalg.norm(blended)
    if norm > 0:
        blended = blended / norm
    return blended


def histogram_similarity(hist_a, hist_b):
    """
    Compare two appearance histograms and return a similarity score.

    OpenCV correlation returns values in a range centered around 0, with higher
    values meaning more similar histograms. We remap that value to an easier
    0-to-1 style score for the reassociation step.
    """
    if hist_a is None or hist_b is None:
        return 0.0

    score = cv2.compareHist(
        hist_a.astype(np.float32),
        hist_b.astype(np.float32),
        cv2.HISTCMP_CORREL
    )
    return float((score + 1.0) / 2.0)


def build_predicted_bbox(last_bbox, velocity):
    """
    Predict where the target bbox should appear in the next frame.

    We use the last known bbox and translate it with the estimated target
    velocity. This predicted box is then used as a spatial reference when the
    original track_id disappears during a crossing or short occlusion.
    """
    if last_bbox is None:
        return None

    vx, vy = velocity
    x1, y1, x2, y2 = last_bbox
    return (
        int(round(x1 + vx)),
        int(round(y1 + vy)),
        int(round(x2 + vx)),
        int(round(y2 + vy)),
    )


def update_target_motion(last_bbox, current_bbox):
    """
    Estimate target motion between two consecutive observations.

    The motion is computed from the displacement between the centers of the
    previous and current bbox. This gives a simple velocity vector that helps
    predict where the target should be if tracking becomes unstable.
    """
    if last_bbox is None or current_bbox is None:
        return (0.0, 0.0)

    last_cx, last_cy = bbox_center(last_bbox)
    curr_cx, curr_cy = bbox_center(current_bbox)
    return (curr_cx - last_cx, curr_cy - last_cy)


def is_target_occluded(target_bbox, tracked_people):
    """
    Check whether the target is likely being occluded by another person.

    If another tracked person overlaps enough with the last known target bbox,
    we treat the situation as a probable crossing/occlusion rather than an
    immediate tracking failure.
    """
    if target_bbox is None:
        return False

    for person in tracked_people:
        other_bbox = person["bbox"]
        if other_bbox == target_bbox:
            continue
        if bbox_iou(target_bbox, other_bbox) >= OCCLUSION_IOU_THRESHOLD:
            return True

    return False


def score_reassociation_candidate(frame, bbox, predicted_bbox, target_hist):
    """
    Score one candidate against the expected target state.

    The returned sub-scores are reused both for fallback reassociation and for
    validating that the currently tracked ID still looks like the original
    target during a crossing.
    """
    if predicted_bbox is None or target_hist is None or bbox is None:
        return None

    predicted_area = max(1, bbox_area(predicted_bbox))
    distance = center_distance(predicted_bbox, bbox)
    if distance > REASSOCIATE_MAX_DISTANCE:
        return None

    candidate_hist = compute_appearance_histogram(frame, bbox)
    appearance_score = histogram_similarity(target_hist, candidate_hist)
    iou_score = bbox_iou(predicted_bbox, bbox)
    area_score = min(bbox_area(bbox), predicted_area) / max(bbox_area(bbox), predicted_area)
    distance_score = max(0.0, 1.0 - (distance / REASSOCIATE_MAX_DISTANCE))

    score = (
        0.30 * appearance_score
        + 0.35 * iou_score
        + 0.25 * distance_score
        + 0.10 * area_score
    )

    return {
        "score": score,
        "appearance_score": appearance_score,
        "iou_score": iou_score,
        "distance_score": distance_score,
        "area_score": area_score,
    }


def find_best_reassociation_candidate(frame, tracked_people, predicted_bbox, target_hist):
    """
    Find the best fallback candidate when the target track_id is lost.

    Each visible person is scored using a mix of appearance similarity,
    overlap with the predicted target position, center distance, and bbox area
    consistency. The function also keeps the second-best score so the caller can
    reject ambiguous situations where two people are almost equally plausible.
    """
    if predicted_bbox is None or target_hist is None:
        return None, 0.0, 0.0

    best_candidate = None
    best_score = 0.0
    second_best_score = 0.0

    for person in tracked_people:
        if person["track_id"] is None:
            continue

        bbox = person["bbox"]
        candidate_metrics = score_reassociation_candidate(frame, bbox, predicted_bbox, target_hist)
        if candidate_metrics is None:
            continue

        score = candidate_metrics["score"]

        if score > best_score:
            second_best_score = best_score
            best_score = score
            best_candidate = person
        elif score > second_best_score:
            second_best_score = score

    return best_candidate, best_score, second_best_score


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    known_encodings = load_known_encodings(ENCODINGS_PATH)
    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    model = YOLO(YOLO_MODEL_PATH).to(device)

    source = args.video if args.video else CAMERA_INDEX
    cap = cv2.VideoCapture(source)

    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        if args.video:
            print(f"Cannot open video: {args.video}")
        else:
            print("Cannot open webcam")
        return

    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or FRAME_WIDTH
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_HEIGHT
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"Cannot open output video for writing: {args.output}")
            cap.release()
            return

    # classic variables
    frame_count = 0
    target_locked = False
    target_track_id = None
    last_target_bbox = None
    target_velocity = (0.0, 0.0)
    target_hist = None
    lost_counter = 0
    reassociate_counter = 0
    post_occlusion_counter = 0
    was_target_occluded = False
    max_lost_frames = 5  # If target ID disappears too long, unlock
    metrics = build_metrics(args.video, device)


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        metrics["frames_processed"] += 1
        previous_target_track_id = target_track_id
        was_locked_before_frame = target_locked

        # -----------------------------
        # 1) YOLO tracking on persons
        # -----------------------------
        results = model.track(
            source=frame,
            persist=True,
            verbose=False,
            device=device,
            half=(device == "cuda"),
            classes=[PERSON_CLASS_ID]
        )

        tracked_people = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
            cls = boxes.cls.cpu().numpy() if boxes.cls is not None else []
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

            for i, box in enumerate(xyxy):
                if int(cls[i]) != PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = map(int, box)
                track_id = int(ids[i]) if ids is not None else None

                tracked_people.append({
                    "track_id": track_id,
                    "bbox": (x1, y1, x2, y2),
                })

        if tracked_people:
            metrics["frames_with_people"] += 1

        # -----------------------------
        # 2) If unlocked, run face recognition sometimes
        # -----------------------------
        recognized_face_boxes = []

        if not target_locked and frame_count == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            for encoding, face_loc in zip(face_encodings, face_locations):
                distances = face_recognition.face_distance(known_encodings, encoding)
                if len(distances) == 0:
                    continue

                best_distance = float(distances.min())

                if best_distance < FACE_TOLERANCE:
                    # Rescale face box from half size to full size.
                    top, right, bottom, left = face_loc
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2

                    recognized_face_boxes.append((left, top, right, bottom))

            # Match recognized face with one tracked person by IoU
            for face_box in recognized_face_boxes:
                best_match_id = None
                best_iou = 0.0

                for person in tracked_people:
                    iou = bbox_iou(face_box, person["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_match_id = person["track_id"]

                # A small overlap is enough because face box is inside person box
                if best_match_id is not None and best_iou > 0.01:
                    target_locked = True
                    target_track_id = best_match_id
                    if not metrics["lock_acquired"]:
                        metrics["lock_acquired"] = True
                        metrics["lock_frame"] = metrics["frames_processed"]
                        metrics["initial_target_track_id"] = best_match_id
                    metrics["times_lock_acquired"] += 1
                    lost_counter = 0
                    reassociate_counter = 0
                    post_occlusion_counter = 0
                    was_target_occluded = False
                    target_velocity = (0.0, 0.0)

                    for person in tracked_people:
                        if person["track_id"] == target_track_id:
                            last_target_bbox = person["bbox"]
                            target_hist = compute_appearance_histogram(frame, last_target_bbox)
                            break
                    break

        frame_count = (frame_count + 1) % RECOGNITION_INTERVAL

        # -----------------------------
        # 3) Follow locked target ID
        # -----------------------------
        target_found_this_frame = False
        target_occluded = False
        target_match_suspect = False
        current_track_score = None
        current_track_appearance = None
        reassociation_score = None
        reassociation_margin = None

        if target_locked and target_track_id is None:
            target_locked = False
            last_target_bbox = None
            target_velocity = (0.0, 0.0)
            target_hist = None
            lost_counter = 0
            reassociate_counter = 0
            post_occlusion_counter = 0
            was_target_occluded = False

        if target_locked and target_track_id is not None:
            predicted_bbox = build_predicted_bbox(last_target_bbox, target_velocity)

            for person in tracked_people:
                if person["track_id"] == target_track_id:
                    current_bbox = person["bbox"]
                    current_occluded = is_target_occluded(current_bbox, tracked_people)
                    current_metrics = score_reassociation_candidate(
                        frame,
                        current_bbox,
                        predicted_bbox if predicted_bbox is not None else current_bbox,
                        target_hist
                    )
                    if current_metrics is not None:
                        current_track_score = current_metrics["score"]
                        current_track_appearance = current_metrics["appearance_score"]

                    if (
                        current_occluded
                        and current_metrics is not None
                        and current_metrics["score"] < TRACK_VALIDATION_MIN_SCORE
                        and current_metrics["appearance_score"] < TRACK_VALIDATION_MIN_APPEARANCE
                    ):
                        target_match_suspect = True
                        target_occluded = True
                    else:
                        target_velocity = update_target_motion(last_target_bbox, current_bbox)
                        last_target_bbox = current_bbox
                        target_hist = update_target_histogram(
                            target_hist,
                            compute_appearance_histogram(frame, current_bbox)
                        )
                        target_found_this_frame = True
                        lost_counter = 0
                        reassociate_counter = 0
                        if not current_occluded and post_occlusion_counter > 0:
                            post_occlusion_counter += 1
                        elif current_occluded:
                            post_occlusion_counter = 0
                            was_target_occluded = True
                    break

            if not target_found_this_frame:
                reassociate_counter += 1
                currently_occluded = is_target_occluded(last_target_bbox, tracked_people)

                if currently_occluded:
                    target_occluded = True
                    post_occlusion_counter = 0
                    was_target_occluded = True
                    candidate = None
                    candidate_score = 0.0
                    second_best_score = 0.0
                    candidate_margin = 0.0
                    reassociation_score = 0.0
                    reassociation_margin = 0.0
                    lost_counter += 1
                else:
                    if was_target_occluded and post_occlusion_counter == 0:
                        post_occlusion_counter = 1
                        was_target_occluded = False
                    elif target_match_suspect or post_occlusion_counter > 0:
                        post_occlusion_counter += 1

                    if (
                        post_occlusion_counter > 0
                        and post_occlusion_counter <= POST_OCCLUSION_REASSOCIATE_FRAMES
                        and reassociate_counter > REASSOCIATE_HOLD_FRAMES
                        and reassociate_counter <= MAX_REASSOCIATE_FRAMES
                        and len(tracked_people) > 0
                    ):
                        candidate, candidate_score, second_best_score = find_best_reassociation_candidate(
                            frame,
                            tracked_people,
                            predicted_bbox,
                            target_hist
                        )
                        candidate_margin = candidate_score - second_best_score
                        reassociation_score = candidate_score
                        reassociation_margin = candidate_margin
                    else:
                        candidate = None
                        candidate_score = 0.0
                        second_best_score = 0.0
                        candidate_margin = 0.0
                        reassociation_score = 0.0
                        reassociation_margin = 0.0

                    if (
                        candidate is not None
                        and candidate_score >= REASSOCIATE_MIN_SCORE
                        and candidate_margin >= REASSOCIATE_MIN_MARGIN
                    ):
                        if candidate["track_id"] != previous_target_track_id:
                            metrics["reassociation_successes"] += 1
                        target_track_id = candidate["track_id"]
                        target_velocity = update_target_motion(last_target_bbox, candidate["bbox"])
                        last_target_bbox = candidate["bbox"]
                        target_hist = update_target_histogram(
                            target_hist,
                            compute_appearance_histogram(frame, candidate["bbox"])
                        )
                        target_found_this_frame = True
                        lost_counter = 0
                        reassociate_counter = 0
                        post_occlusion_counter = 0
                    else:
                        lost_counter += 1

                if lost_counter > max_lost_frames:
                    target_locked = False
                    target_track_id = None
                    last_target_bbox = None
                    target_velocity = (0.0, 0.0)
                    target_hist = None
                    lost_counter = 0
                    reassociate_counter = 0
                    post_occlusion_counter = 0
                    was_target_occluded = False

        if target_occluded:
            metrics["occlusion_frames"] += 1

        if target_locked:
            metrics["frames_locked"] += 1

        if target_found_this_frame:
            metrics["frames_target_visible"] += 1

        if lost_counter > metrics["max_consecutive_lost_frames"]:
            metrics["max_consecutive_lost_frames"] = lost_counter

        if was_locked_before_frame and not target_locked:
            metrics["times_lock_lost"] += 1

        # -----------------------------
        # 4) Draw everything
        # -----------------------------
        for person in tracked_people:
            if person["track_id"] is None:
                continue
            x1, y1, x2, y2 = person["bbox"]
            tid = person["track_id"]

            if target_locked and tid == target_track_id:
                color = (0, 255, 0)
                label = f"Youri | ID {tid}"
            else:
                color = (0, 0, 255)
                label = f"Person | ID {tid}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, label, (x1, max(20, y1 - 10)), color)

        # Optional: draw recognized face boxes while unlocking
        if not target_locked:
            for left, top, right, bottom in recognized_face_boxes:
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
                draw_label(frame, "Face match", (left, max(20, top - 10)), (255, 255, 0))

        # State text
        if target_locked and target_track_id is not None:
            draw_label(frame, f"LOCKED ON ID {target_track_id}", (10, 30), (0, 255, 0))
            if target_occluded:
                draw_label(frame, "OCCLUSION: REASSOCIATING...", (10, 60), (0, 255, 255))
            elif lost_counter > 0:
                draw_label(frame, f"TARGET TEMP LOST: {lost_counter}", (10, 60), (0, 255, 255))
            if current_track_score is not None:
                draw_label(
                    frame,
                    f"Track score: {current_track_score:.2f} | appearance: {current_track_appearance:.2f}",
                    (10, 90),
                    (255, 255, 255)
                )
            if reassociation_score is not None and lost_counter > 0:
                draw_label(
                    frame,
                    f"Reassoc score: {reassociation_score:.2f} | margin: {reassociation_margin:.2f}",
                    (10, 120),
                    (0, 255, 255)
                )
        else:
            draw_label(frame, "SEARCHING YOURI...", (10, 30), (0, 0, 255))

        if not args.headless:
            display_frame = resize_for_display(
                frame,
                args.display_max_width,
                args.display_max_height,
            )
            cv2.imshow("Lock and Track", display_frame)
        if writer is not None:
            writer.write(frame)

        if not args.headless:
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        if args.max_frames is not None and metrics["frames_processed"] >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()

    metrics["final_target_track_id"] = target_track_id
    metrics["target_present_at_end"] = target_found_this_frame
    metrics["ended_locked"] = target_locked
    metrics = finalize_metrics(metrics)

    if args.metrics_output:
        save_metrics(metrics, args.metrics_output)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
