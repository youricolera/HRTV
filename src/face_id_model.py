import os
import pickle
from pathlib import Path

import cv2
import face_recognition
import numpy as np
import dlib
import torch


# ---------- Config ----------
DATASET_DIR = Path("dataset_faces/youri")
INPUT_PKL = Path("youri_encodings.pkl")
OUTPUT_PKL = Path("youri_encodings_enriched.pkl")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Retry strategies:
# (upsample, model_name, scale_factor, rotation_angle)
STRATEGIES = [
    (1, "hog", 1.0, 0),
    (2, "hog", 1.0, 0),
    (3, "hog", 1.0, 0),
    (1, "hog", 1.5, 0),
    (2, "hog", 1.5, 0),
    (2, "hog", 2.0, 0),
    (2, "hog", 1.5, -10),
    (2, "hog", 1.5, 10),
]


def load_existing_encodings(pkl_path: Path) -> list:
    if not pkl_path.exists():
        return []
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def save_encodings(encodings: list, pkl_path: Path) -> None:
    with open(pkl_path, "wb") as f:
        pickle.dump(encodings, f)


def resize_image(image: np.ndarray, scale_factor: float, max_width: int = 1200) -> np.ndarray:
    h, w = image.shape[:2]

    # resize demandé par la stratégie
    if scale_factor != 1.0:
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # limite de taille pour éviter explosion mémoire
    h, w = image.shape[:2]

    if w > max_width:
        ratio = max_width / w
        new_w = int(w * ratio)
        new_h = int(h * ratio)

        print(f"      -> resizing large image to {new_w}x{new_h}")

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    # Mild CLAHE on luminance channel
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def try_extract_encoding(image_rgb: np.ndarray):
    """
    Try multiple strategies to get one face encoding from an RGB image.
    Returns (encoding or None, strategy_used or None).
    """
    variants = [
        ("original", image_rgb),
        ("contrast", enhance_contrast(image_rgb)),
    ]

    for variant_name, variant in variants:
        for upsample, model_name, scale_factor, angle in STRATEGIES:
            print(
                f"   Trying -> variant={variant_name}, "
                f"model={model_name}, "
                f"upsample={upsample}, "
                f"scale={scale_factor}, "
                f"angle={angle}"
            )

            trial = resize_image(variant, scale_factor)
            trial = rotate_image(trial, angle)

            try:
                print("      -> calling face_locations")
                face_locations = face_recognition.face_locations(
                    trial,
                    number_of_times_to_upsample=upsample,
                    model=model_name,
                )
                print(f"      -> face_locations done, found {len(face_locations)} face(s)")

            except Exception as e:
                print("      ERROR:", e)
                continue

            if len(face_locations) == 0:
                continue

            try:
                print("      -> calling face_encodings")
                encodings = face_recognition.face_encodings(
                    trial,
                    known_face_locations=face_locations,
                    num_jitters=3,
                )
                print(f"      -> face_encodings done, found {len(encodings)} encoding(s)")
            except Exception as e:
                print("      ERROR:", e)
                continue

            if len(encodings) == 0:
                continue

            strategy = {
                "variant": variant_name,
                "upsample": upsample,
                "model": model_name,
                "scale_factor": scale_factor,
                "angle": angle,
            }
            return encodings[0], strategy

    return None, None


def main():
    existing_encodings = load_existing_encodings(INPUT_PKL)
    print(f"Existing encodings loaded: {len(existing_encodings)}")

    added_count = 0
    failed_files = []

    image_files = sorted(
        [
            p for p in DATASET_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    for image_path in image_files:
        print(f"\nProcessing: {image_path.name}")

        try:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print("  -> Cannot read image")
                failed_files.append(image_path.name)
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # First quick standard attempt
            quick_locations = face_recognition.face_locations(
                image_rgb,
                number_of_times_to_upsample=1,
                model="hog",
            )
            quick_encodings = face_recognition.face_encodings(
                image_rgb,
                known_face_locations=quick_locations,
                num_jitters=1,
            ) if len(quick_locations) > 0 else []

            if len(quick_encodings) > 0:
                print("  -> Already detectable with standard settings, skipped")
                continue

            encoding, strategy = try_extract_encoding(image_rgb)

            if encoding is not None:
                existing_encodings.append(encoding)
                added_count += 1
                print(f"  -> Recovered with strategy: {strategy}")
            else:
                print("  -> Still no face detected")
                failed_files.append(image_path.name)

        except Exception as e:
            print(f"  -> Error: {e}")
            failed_files.append(image_path.name)

    save_encodings(existing_encodings, OUTPUT_PKL)

    print("\n===== Summary =====")
    print(f"New encodings added: {added_count}")
    print(f"Total encodings saved: {len(existing_encodings)}")
    print(f"Output file: {OUTPUT_PKL}")

    if failed_files:
        print("\nImages still failing:")
        for name in failed_files:
            print(f" - {name}")


if __name__ == "__main__":
    main()