from __future__ import annotations

import base64
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from geometry_utils import (
    asymmetry_index,
    euclidean_distance,
    line_angle_degrees,
    line_slope,
    mean_point,
    isotropic_normalize_landmarks,
    normalize_line_angle,
    remove_head_roll,
    weaker_side,
)

from hb_rules import classify_house_brackmann
from regional_assessment import (
    REGIONAL_ASSESSMENT_VERSION,
    assess_regions,
    calculate_face_function_score,
    infer_affected_side,
)


# ============================================================
# KONFIGURASI DASAR
# ============================================================

EPSILON = 1e-8

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model MediaPipe tidak ditemukan: {MODEL_PATH}"
    )


# ============================================================
# LANDMARK MATA DAN ALIS
# ============================================================

RIGHT_EYE_POINTS = (33, 133, 159, 145)
LEFT_EYE_POINTS = (362, 263, 386, 374)

RIGHT_BROW_POINTS = (70, 63, 105, 66, 107)
LEFT_BROW_POINTS = (336, 296, 334, 293, 300)

RIGHT_EYE_VERTICAL_PAIRS = (
    (159, 145),
    (158, 153),
    (160, 144),
)

LEFT_EYE_VERTICAL_PAIRS = (
    (386, 374),
    (385, 380),
    (387, 373),
)

RIGHT_EYE_HORIZONTAL = (33, 133)
LEFT_EYE_HORIZONTAL = (362, 263)


# ============================================================
# LANDMARK MULUT
# ============================================================

# Beberapa landmark dipakai sebagai satu kelompok agar hasil lebih stabil
# daripada hanya bergantung pada satu landmark di setiap sisi.
MOUTH_CORNER_GROUPS = (
    (61, 146, 185),
    (291, 375, 409),
)

# Kelompok sisi bibir atas untuk gerakan snarl.
UPPER_LIP_SIDE_GROUPS = (
    (37, 39, 40),
    (267, 269, 270),
)


# ============================================================
# PROTOKOL GERAKAN DAN QUALITY GATE
# ============================================================

REQUIRED_MOVEMENTS = (
    "diam",
    "angkat_alis",
    "tutup_mata",
    "senyum",
    "mencucu",
)

MINIMUM_VALID_FRAMES = 4
MAXIMUM_FRAMES_PER_MOVEMENT = 6

# Threshold ini hanya memeriksa apakah gerakan dilakukan atau tidak.
# Nilainya merupakan threshold kualitas pilot, bukan threshold klinis.
# MIN_SNARL_MAX_EXCURSION = 0.015
MIN_PUCKER_MAX_EXCURSION = 0.015
MIN_PUCKER_WIDTH_REDUCTION = 0.05


# ============================================================
# MEDIAPIPE FACE LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH),
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    # Mengizinkan maksimal dua wajah agar backend dapat menolak frame
    # yang berisi lebih dari satu wajah.
    num_faces=2,
    min_face_detection_confidence=0.7,
    min_face_presence_confidence=0.7,
)

landmarker = vision.FaceLandmarker.create_from_options(options)


# ============================================================
# UTILITAS GAMBAR DAN LANDMARK
# ============================================================

def decode_base64_image(base64_string: str) -> np.ndarray | None:
    try:
        if not isinstance(base64_string, str):
            return None

        if "," in base64_string:
            base64_string = base64_string.split(",", maxsplit=1)[1]

        image_bytes = base64.b64decode(
            base64_string,
            validate=True,
        )

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        return cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

    except Exception as error:
        print("DECODE IMAGE ERROR:", error)
        return None


def detect_landmarks(image_bgr: np.ndarray) -> list[Any] | None:
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb,
    )

    detection_result = landmarker.detect(mp_image)

    # Frame hanya diterima jika tepat satu wajah terdeteksi.
    if len(detection_result.face_landmarks) != 1:
        return None

    return detection_result.face_landmarks[0]


def eye_aperture_ratio(
    points: np.ndarray,
    vertical_pairs: tuple[tuple[int, int], ...],
    horizontal_pair: tuple[int, int],
) -> float:
    vertical_distances = [
        euclidean_distance(
            points[top_index],
            points[bottom_index],
        )
        for top_index, bottom_index in vertical_pairs
    ]

    mean_vertical = float(np.mean(vertical_distances))

    horizontal_distance = euclidean_distance(
        points[horizontal_pair[0]],
        points[horizontal_pair[1]],
    )

    return mean_vertical / (horizontal_distance + EPSILON)


def ordered_group_centers(
    points: np.ndarray,
    group_a: tuple[int, ...],
    group_b: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mengambil rata-rata beberapa landmark lalu mengurutkannya
    sebagai sisi kiri dan kanan gambar berdasarkan koordinat x.
    """
    center_a = mean_point(points, group_a)
    center_b = mean_point(points, group_b)

    centers = [center_a, center_b]
    centers.sort(key=lambda point: float(point[0]))

    return centers[0], centers[1]


# ============================================================
# EKSTRAKSI GEOMETRI SATU FRAME
# ============================================================

def extract_frame_geometry(
    landmarks: list[Any],
    image_shape: tuple[int, ...],
) -> dict[str, float]:
    image_height, image_width = image_shape[:2]

    # Sudut dihitung dari koordinat piksel supaya rasio aspek gambar
    # tidak mengubah hasil angle.
    raw_points = np.array(
        [
            [
                landmark.x * image_width,
                landmark.y * image_height,
            ]
            for landmark in landmarks
        ],
        dtype=np.float64,
    )

    groups = [
        {
            "eye_points": RIGHT_EYE_POINTS,
            "brow_points": RIGHT_BROW_POINTS,
            "vertical_pairs": RIGHT_EYE_VERTICAL_PAIRS,
            "horizontal_pair": RIGHT_EYE_HORIZONTAL,
        },
        {
            "eye_points": LEFT_EYE_POINTS,
            "brow_points": LEFT_BROW_POINTS,
            "vertical_pairs": LEFT_EYE_VERTICAL_PAIRS,
            "horizontal_pair": LEFT_EYE_HORIZONTAL,
        },
    ]

    for group in groups:
        group["raw_eye_center"] = mean_point(
            raw_points,
            group["eye_points"],
        )

    # left/right pada backend berarti kiri/kanan gambar.
    groups.sort(
        key=lambda group: float(group["raw_eye_center"][0])
    )

    image_left_group = groups[0]
    image_right_group = groups[1]

    rotated_points, head_roll = remove_head_roll(
        raw_points,
        image_left_group["raw_eye_center"],
        image_right_group["raw_eye_center"],
    )

    if abs(head_roll) > 20.0:
        raise ValueError("pose_rejected")

    # Normalisasi dilakukan setelah koreksi head-roll menggunakan jarak
    # antar-mata sebagai satu skala fisik yang konsisten. Min-max scaling
    # berdasarkan bounding box tiap frame dapat berubah ketika mulut/alis
    # bergerak dan membuat excursion ikut berubah secara artifisial.
    rotated_left_eye_center = mean_point(
        rotated_points,
        image_left_group["eye_points"],
    )
    rotated_right_eye_center = mean_point(
        rotated_points,
        image_right_group["eye_points"],
    )

    scaled_points = isotropic_normalize_landmarks(
        rotated_points,
        rotated_left_eye_center,
        rotated_right_eye_center,
    )

    left_eye_center = mean_point(
        scaled_points,
        image_left_group["eye_points"],
    )
    right_eye_center = mean_point(
        scaled_points,
        image_right_group["eye_points"],
    )

    left_brow_center = mean_point(
        scaled_points,
        image_left_group["brow_points"],
    )
    right_brow_center = mean_point(
        scaled_points,
        image_right_group["brow_points"],
    )

    left_eye_aperture = eye_aperture_ratio(
        scaled_points,
        image_left_group["vertical_pairs"],
        image_left_group["horizontal_pair"],
    )
    right_eye_aperture = eye_aperture_ratio(
        scaled_points,
        image_right_group["vertical_pairs"],
        image_right_group["horizontal_pair"],
    )

    left_brow_eye_distance = euclidean_distance(
        left_brow_center,
        left_eye_center,
    )
    right_brow_eye_distance = euclidean_distance(
        right_brow_center,
        right_eye_center,
    )

    # Kelompok sudut bibir pada koordinat hasil Min-Max.
    mouth_left, mouth_right = ordered_group_centers(
        scaled_points,
        MOUTH_CORNER_GROUPS[0],
        MOUTH_CORNER_GROUPS[1],
    )

    # Kelompok sudut bibir pada koordinat piksel yang telah dikoreksi roll.
    raw_mouth_left, raw_mouth_right = ordered_group_centers(
        rotated_points,
        MOUTH_CORNER_GROUPS[0],
        MOUTH_CORNER_GROUPS[1],
    )

    mouth_width = euclidean_distance(
        mouth_left,
        mouth_right,
    )

    mouth_angle = normalize_line_angle(
        line_angle_degrees(
            raw_mouth_left,
            raw_mouth_right,
        )
    )

    mouth_slope = line_slope(
        raw_mouth_left,
        raw_mouth_right,
    )

    rotated_left_brow = mean_point(
        rotated_points,
        image_left_group["brow_points"],
    )
    rotated_right_brow = mean_point(
        rotated_points,
        image_right_group["brow_points"],
    )

    brow_angle = normalize_line_angle(
        line_angle_degrees(
            rotated_left_brow,
            rotated_right_brow,
        )
    )

    upper_left, upper_right = ordered_group_centers(
        scaled_points,
        UPPER_LIP_SIDE_GROUPS[0],
        UPPER_LIP_SIDE_GROUPS[1],
    )

    return {
        "head_roll_deg": float(head_roll),

        "left_eye_aperture": float(left_eye_aperture),
        "right_eye_aperture": float(right_eye_aperture),

        "left_brow_eye_distance": float(left_brow_eye_distance),
        "right_brow_eye_distance": float(right_brow_eye_distance),

        "left_mouth_x": float(mouth_left[0]),
        "right_mouth_x": float(mouth_right[0]),
        "left_mouth_y": float(mouth_left[1]),
        "right_mouth_y": float(mouth_right[1]),
        "mouth_width": float(mouth_width),

        "left_upper_lip_y": float(upper_left[1]),
        "right_upper_lip_y": float(upper_right[1]),

        "mouth_angle_deg": float(mouth_angle),
        "mouth_slope": float(mouth_slope),
        "brow_angle_deg": float(brow_angle),
    }


# ============================================================
# AGREGASI DAN FITUR ANTAR-GERAKAN
# ============================================================

def aggregate_measurements(
    measurements: list[dict[str, float]],
) -> dict[str, float]:
    if not measurements:
        raise ValueError("insufficient_valid_frames")

    keys = measurements[0].keys()

    return {
        key: float(
            median(
                measurement[key]
                for measurement in measurements
            )
        )
        for key in keys
    }


def angle_difference(
    active_angle: float,
    neutral_angle: float,
) -> float:
    return abs(
        normalize_line_angle(
            active_angle - neutral_angle
        )
    )


def build_movement_features(
    aggregated: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | str]]:
    rest = aggregated["diam"]
    brow = aggregated["angkat_alis"]
    eye = aggregated["tutup_mata"]
    smile = aggregated["senyum"]
    # snarl = aggregated["snarl"]
    pucker = aggregated["mencucu"]

    rest_mouth_width = max(
        rest["mouth_width"],
        EPSILON,
    )

    # Angkat alis: jarak alis-mata seharusnya bertambah.
    left_brow_excursion = max(
        0.0,
        brow["left_brow_eye_distance"]
        - rest["left_brow_eye_distance"],
    )
    right_brow_excursion = max(
        0.0,
        brow["right_brow_eye_distance"]
        - rest["right_brow_eye_distance"],
    )

    # Tutup mata: 1 berarti menutup penuh relatif terhadap kondisi diam.
    left_eye_closure = float(
        np.clip(
            1.0
            - eye["left_eye_aperture"]
            / (rest["left_eye_aperture"] + EPSILON),
            0.0,
            1.0,
        )
    )
    right_eye_closure = float(
        np.clip(
            1.0
            - eye["right_eye_aperture"]
            / (rest["right_eye_aperture"] + EPSILON),
            0.0,
            1.0,
        )
    )

    # Senyum: sudut bibir bergerak ke atas relatif terhadap diam.
    left_smile_excursion = max(
        0.0,
        rest["left_mouth_y"] - smile["left_mouth_y"],
    )
    right_smile_excursion = max(
        0.0,
        rest["right_mouth_y"] - smile["right_mouth_y"],
    )

    # Snarl: sisi bibir atas bergerak ke atas.
    # Dibagi lebar mulut saat diam agar lebih stabil terhadap ukuran wajah.
    # left_snarl_excursion = (
    #     max(
    #         0.0,
    #         rest["left_upper_lip_y"]
    #         - snarl["left_upper_lip_y"],
    #     )
    #     / rest_mouth_width
    # )
    # right_snarl_excursion = (
    #     max(
    #         0.0,
    #         rest["right_upper_lip_y"]
    #         - snarl["right_upper_lip_y"],
    #     )
    #     / rest_mouth_width
    # )

    # Mencucu: kedua sudut bibir bergerak menuju tengah.
    # Perpindahan dibagi lebar mulut saat diam.
    left_pucker_excursion = (
        max(
            0.0,
            pucker["left_mouth_x"]
            - rest["left_mouth_x"],
        )
        / rest_mouth_width
    )
    right_pucker_excursion = (
        max(
            0.0,
            rest["right_mouth_x"]
            - pucker["right_mouth_x"],
        )
        / rest_mouth_width
    )

    pucker_width_reduction = float(
        np.clip(
            1.0
            - pucker["mouth_width"]
            / rest_mouth_width,
            0.0,
            1.0,
        )
    )

    return {
        "rest": {
            "eye_delta": abs(
                rest["left_eye_aperture"]
                - rest["right_eye_aperture"]
            ),
            "eye_ai": asymmetry_index(
                rest["left_eye_aperture"],
                rest["right_eye_aperture"],
            ),
            "brow_ai": asymmetry_index(
                rest["left_brow_eye_distance"],
                rest["right_brow_eye_distance"],
            ),
            "mouth_angle_deg": abs(rest["mouth_angle_deg"]),
            "mouth_angle_signed_deg": rest["mouth_angle_deg"],
            "mouth_slope": rest["mouth_slope"],
        },

        "angkat_alis": {
            "left": left_brow_excursion,
            "right": right_brow_excursion,
            "asymmetry_index": asymmetry_index(
                left_brow_excursion,
                right_brow_excursion,
                activity_floor=0.005,
            ),
            "angle_change_deg": angle_difference(
                brow["brow_angle_deg"],
                rest["brow_angle_deg"],
            ),
            "strength": (
                left_brow_excursion
                + right_brow_excursion
            ) / 2.0,
            "weaker_side": weaker_side(
                left_brow_excursion,
                right_brow_excursion,
            ),
        },

        "tutup_mata": {
            "left": left_eye_closure,
            "right": right_eye_closure,
            "asymmetry_index": asymmetry_index(
                left_eye_closure,
                right_eye_closure,
                activity_floor=0.05,
            ),
            "minimum_closure": min(
                left_eye_closure,
                right_eye_closure,
            ),
            "strength": (
                left_eye_closure
                + right_eye_closure
            ) / 2.0,
            "weaker_side": weaker_side(
                left_eye_closure,
                right_eye_closure,
            ),
        },

        "senyum": {
            "left": left_smile_excursion,
            "right": right_smile_excursion,
            "asymmetry_index": asymmetry_index(
                left_smile_excursion,
                right_smile_excursion,
                activity_floor=0.005,
            ),
            "angle_change_deg": angle_difference(
                smile["mouth_angle_deg"],
                rest["mouth_angle_deg"],
            ),
            "mouth_angle_deg": smile["mouth_angle_deg"],
            "mouth_slope": smile["mouth_slope"],
            "strength": (
                left_smile_excursion
                + right_smile_excursion
            ) / 2.0,
            "weaker_side": weaker_side(
                left_smile_excursion,
                right_smile_excursion,
            ),
        },

        # "snarl": {
        #     "left": left_snarl_excursion,
        #     "right": right_snarl_excursion,
        #     "asymmetry_index": asymmetry_index(
        #         left_snarl_excursion,
        #         right_snarl_excursion,
        #         activity_floor=0.005,
        #     ),
        #     "strength": (
        #         left_snarl_excursion
        #         + right_snarl_excursion
        #     ) / 2.0,
        #     "max_excursion": max(
        #         left_snarl_excursion,
        #         right_snarl_excursion,
        #     ),
        #     "weaker_side": weaker_side(
        #         left_snarl_excursion,
        #         right_snarl_excursion,
        #     ),
        # },

        "mencucu": {
            "left": left_pucker_excursion,
            "right": right_pucker_excursion,
            "asymmetry_index": asymmetry_index(
                left_pucker_excursion,
                right_pucker_excursion,
                activity_floor=0.005,
            ),
            "strength": (
                left_pucker_excursion
                + right_pucker_excursion
            ) / 2.0,
            "max_excursion": max(
                left_pucker_excursion,
                right_pucker_excursion,
            ),
            "width_reduction": pucker_width_reduction,
            "weaker_side": weaker_side(
                left_pucker_excursion,
                right_pucker_excursion,
            ),
        },
    }



# ============================================================
# PENILAIAN REGIONAL DAN RULE HOUSE-BRACKMANN
# ============================================================
#
# Implementasi dipisahkan:
# - regional_assessment.py:
#   menerjemahkan fitur geometris menjadi kategori regional.
# - hb_rules.py:
#   memetakan kategori regional menjadi Tingkat I-VI.
#
# Grade tidak ditentukan langsung dari satu nilai asimetri.


# ============================================================
# ANALISIS SELURUH FRAME
# ============================================================

# Quality gate hanya untuk kegagalan teknis.
# Gerakan kecil, mata tidak menutup, atau mulut tidak bergerak
# tetap diterima sebagai data untuk klasifikasi.
def analyze_detection_frames(frames: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(frames, dict):
        return {
            "quality_status": "invalid_frames_structure",
        }

    measurements_by_movement: dict[str, dict[str, float]] = {}
    valid_frame_count: dict[str, int] = {}
    rejected_frame_count: dict[str, int] = {}

    for movement in REQUIRED_MOVEMENTS:
        frame_list = frames.get(movement)

        if not isinstance(frame_list, list):
            return {
                "quality_status": "missing_movement",
                "movement": movement,
            }

        if len(frame_list) < MINIMUM_VALID_FRAMES:
            return {
                "quality_status": "insufficient_received_frames",
                "movement": movement,
                "received_frame_count": len(frame_list),
            }

        if len(frame_list) > MAXIMUM_FRAMES_PER_MOVEMENT:
            return {
                "quality_status": "too_many_frames",
                "movement": movement,
                "received_frame_count": len(frame_list),
                "maximum_frame_count": MAXIMUM_FRAMES_PER_MOVEMENT,
            }

        valid_measurements: list[dict[str, float]] = []
        rejected_count = 0

        for frame_base64 in frame_list:
            try:
                image = decode_base64_image(frame_base64)

                if image is None:
                    rejected_count += 1
                    continue

                landmarks = detect_landmarks(image)

                if landmarks is None:
                    rejected_count += 1
                    continue

                measurement = extract_frame_geometry(
                    landmarks,
                    image.shape,
                )

                valid_measurements.append(measurement)

            except Exception as error:
                rejected_count += 1
                print(
                    f"FRAME REJECTED {movement}: {error}"
                )

        valid_frame_count[movement] = len(valid_measurements)
        rejected_frame_count[movement] = rejected_count

        if len(valid_measurements) < MINIMUM_VALID_FRAMES:
            return {
                "quality_status": "insufficient_valid_frames",
                "movement": movement,
                "valid_frame_count": valid_frame_count,
                "rejected_frame_count": rejected_frame_count,
            }

        measurements_by_movement[movement] = aggregate_measurements(
            valid_measurements
        )

    feature_values = build_movement_features(
        measurements_by_movement
    )

    # Gerakan dengan excursion rendah tidak langsung ditolak karena
    # kondisi tersebut dapat merupakan disfungsi berat, bukan sekadar
    # pengguna tidak mengikuti instruksi. Sistem hanya memberi warning.
    quality_warnings: list[str] = []

    # snarl_feature = feature_values["snarl"]
    pucker_feature = feature_values["mencucu"]

    # if (
    #     float(snarl_feature["max_excursion"])
    #     < MIN_SNARL_MAX_EXCURSION
    # ):
    #     quality_warnings.append("snarl_low_excursion")

    if (
        float(pucker_feature["max_excursion"])
        < MIN_PUCKER_MAX_EXCURSION
        and float(pucker_feature["width_reduction"])
        < MIN_PUCKER_WIDTH_REDUCTION
    ):
        quality_warnings.append("pucker_low_excursion")

    regional_result = assess_regions(feature_values)
    hb_result = classify_house_brackmann(regional_result)

    # Persentase dihitung dari kategori regional yang sama dan
    # hanya digunakan sebagai skor progres/monitoring.
    face_function_score = calculate_face_function_score(
        regional_result,
        hb_result.get("grade"),
    )

    affected_side = infer_affected_side(regional_result)

    classification_status = hb_result["status"]

    result = {
        "quality_status": "valid",
        "classification_status": classification_status,
        "hb_grade": hb_result["grade"],
        "hb_roman": hb_result["roman"],
        "hb_label": hb_result["label"],
        "hb_matched_rule": hb_result["matched_rule"],
        "classification_message": hb_result["message"],
        "inconsistent_regions": hb_result.get(
            "inconsistent_regions",
            [],
        ),
        "face_function_score": face_function_score,
        "regional_result": regional_result,
        "feature_values": feature_values,
        "affected_side": affected_side,
        "algorithm_version": "single-concept-relaxed-quality-gate-v2",
        "threshold_version": REGIONAL_ASSESSMENT_VERSION,
        "quality_warnings": quality_warnings,
        "valid_frame_count": valid_frame_count,
        "rejected_frame_count": rejected_frame_count,
        "side_reference": "image",
    }

    # Data teknis valid, tetapi pola regional tidak konsisten.
    # Jangan paksa mengeluarkan Tingkat I-VI.
    if classification_status == "repeat_required":
        result["message"] = hb_result["message"]
        result["invalid_movements"] = hb_result.get(
            "inconsistent_regions",
            [],
        )

    return result
