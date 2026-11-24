from __future__ import annotations

import base64
from pathlib import Path
from statistics import median
from typing import Any, Callable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


EPSILON = 1e-8

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

ALGORITHM_VERSION = "therapy-realtime-mediapipe-v1"
THRESHOLD_VERSION = "therapy-system-v1-fixed"

# ============================================================
# THRESHOLD KUALITAS PENGAMBILAN
# ============================================================

MIN_FACE_DETECTION_CONFIDENCE = 0.60
MIN_FACE_PRESENCE_CONFIDENCE = 0.60

MAX_HEAD_ROLL_DEG = 22.0
MAX_FRONTAL_OFFSET = 0.24

MIN_VALID_NEUTRAL_FRAMES = 1
MIN_VALID_ACTION_FRAMES = 5
MIN_VALID_ACTION_RATIO = 0.50

MIN_CONSECUTIVE_DETECTED_FRAMES = 3
MIN_DETECTED_FRAME_RATIO = 0.60

# ============================================================
# THRESHOLD GERAKAN SISTEM V1
#
# Nilai ini dikunci agar implementasi konsisten.
# Ini adalah threshold sistem/prototipe, bukan threshold klinis.
# ============================================================

THRESHOLDS: dict[str, dict[str, float]] = {
    "cheek_puff": {
        "max_lip_gap_ratio": 0.22,
        "min_blend_change": 0.06,
        "min_blend_action": 0.15,
        "min_geometry_change": 0.025,
    },
    "pucker": {
        "min_pucker_change": 0.08,
        "min_width_decrease": 0.025,
        "alt_min_pucker_change": 0.13,
        "alt_min_funnel_change": 0.04,
    },
    "blowing_shape": {
        "min_pucker_change": 0.06,
        "min_funnel_change": 0.04,
        "max_jaw_open": 0.60,
    },
    "lip_stretch": {
        "min_stretch_change": 0.07,
        "min_width_increase": 0.025,
    },
    "eye_close": {
        "min_strongest_closure": 0.38,
        "min_weakest_closure": 0.14,
    },
    "brow_frown": {
        "min_brow_down_change": 0.06,
        "min_inner_brow_decrease": 0.008,
    },
    "blink": {
        "min_closed_value": 0.35,
        "min_reopen_ratio": 0.60,
    },
    "smile_eye_close": {
        "min_smile_change": 0.06,
        "min_eye_closure": 0.30,
    },
}

SUPPORTED_MOVEMENTS = {
    "cheek_puff",
    "pucker",
    "blowing_shape",
    "lip_stretch",
    "eye_close",
    "brow_frown",
    "eye_close_lip_stretch",
    "blink",
    "smile_eye_close",
}


class TherapyQualityError(ValueError):
    def __init__(
        self,
        message: str,
        quality_status: str = "invalid_capture",
    ) -> None:
        super().__init__(message)
        self.quality_status = quality_status


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model MediaPipe tidak ditemukan: {MODEL_PATH}"
    )


_base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH),
)

_options = vision.FaceLandmarkerOptions(
    base_options=_base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=2,
    min_face_detection_confidence=(
        MIN_FACE_DETECTION_CONFIDENCE
    ),
    min_face_presence_confidence=(
        MIN_FACE_PRESENCE_CONFIDENCE
    ),
    output_face_blendshapes=True,
)

_landmarker = vision.FaceLandmarker.create_from_options(
    _options
)


def _decode_base64_image(
    encoded: str,
) -> np.ndarray:
    if not isinstance(encoded, str) or not encoded:
        raise TherapyQualityError(
            "Frame kamera kosong."
        )

    if "," in encoded:
        encoded = encoded.split(",", maxsplit=1)[1]

    try:
        image_bytes = base64.b64decode(
            encoded,
            validate=True,
        )
    except Exception as error:
        raise TherapyQualityError(
            "Frame Base64 tidak dapat dibaca."
        ) from error

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise TherapyQualityError(
            "Frame kamera tidak dapat didekode."
        )

    return image


def _distance(
    points: np.ndarray,
    index_a: int,
    index_b: int,
) -> float:
    return float(
        np.linalg.norm(
            points[index_a, :2]
            - points[index_b, :2]
        )
    )


def _eye_aperture(
    points: np.ndarray,
    vertical_pairs: tuple[
        tuple[int, int],
        ...,
    ],
    horizontal_pair: tuple[int, int],
) -> float:
    vertical = float(
        np.mean(
            [
                _distance(points, top, bottom)
                for top, bottom
                in vertical_pairs
            ]
        )
    )

    horizontal = _distance(
        points,
        horizontal_pair[0],
        horizontal_pair[1],
    )

    return vertical / (
        horizontal + EPSILON
    )


def _detect_with_rotation(
    image_bgr: np.ndarray,
) -> Any:
    candidates = [
        image_bgr,
        cv2.rotate(
            image_bgr,
            cv2.ROTATE_90_CLOCKWISE,
        ),
        cv2.rotate(
            image_bgr,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ),
        cv2.rotate(
            image_bgr,
            cv2.ROTATE_180,
        ),
    ]

    multiple_faces_found = False

    for candidate in candidates:
        image_rgb = cv2.cvtColor(
            candidate,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb,
        )

        result = _landmarker.detect(mp_image)

        if len(result.face_landmarks) == 1:
            return result

        if len(result.face_landmarks) > 1:
            multiple_faces_found = True

    if multiple_faces_found:
        raise TherapyQualityError(
            "Pastikan hanya satu wajah berada di depan kamera.",
            "multiple_faces",
        )

    raise TherapyQualityError(
        "Wajah belum terbaca dengan jelas.",
        "face_not_detected",
    )


def _extract_frame_metrics(
    encoded: str,
) -> dict[str, float]:
    image_bgr = _decode_base64_image(encoded)
    result = _detect_with_rotation(image_bgr)

    points = np.array(
        [
            [
                float(item.x),
                float(item.y),
                float(item.z),
            ]
            for item in result.face_landmarks[0]
        ],
        dtype=np.float64,
    )

    interocular = max(
        _distance(points, 33, 263),
        EPSILON,
    )

    eye_line_dx = (
        points[263, 0]
        - points[33, 0]
    )

    eye_line_dy = (
        points[263, 1]
        - points[33, 1]
    )

    head_roll = float(
        np.degrees(
            np.arctan2(
                eye_line_dy,
                eye_line_dx,
            )
        )
    )

    eye_mid_x = (
        points[33, 0]
        + points[263, 0]
    ) / 2.0

    frontal_offset = abs(
        points[1, 0]
        - eye_mid_x
    ) / interocular

    if abs(head_roll) > MAX_HEAD_ROLL_DEG:
        raise TherapyQualityError(
            "Kepala terlalu miring.",
            "head_tilt",
        )

    if frontal_offset > MAX_FRONTAL_OFFSET:
        raise TherapyQualityError(
            "Hadapkan wajah lebih lurus ke kamera.",
            "not_frontal",
        )

    face_width = max(
        _distance(points, 234, 454),
        EPSILON,
    )

    mouth_width = (
        _distance(points, 61, 291)
        / face_width
    )

    lip_gap_ratio = (
        _distance(points, 13, 14)
        / max(
            _distance(points, 61, 291),
            EPSILON,
        )
    )

    left_eye_aperture = _eye_aperture(
        points,
        (
            (159, 145),
            (158, 153),
            (160, 144),
        ),
        (33, 133),
    )

    right_eye_aperture = _eye_aperture(
        points,
        (
            (386, 374),
            (385, 380),
            (387, 373),
        ),
        (362, 263),
    )

    inner_brow_distance = (
        _distance(points, 107, 336)
        / face_width
    )

    left_cheek_to_mouth = (
        _distance(points, 234, 61)
        / face_width
    )

    right_cheek_to_mouth = (
        _distance(points, 454, 291)
        / face_width
    )

    scores: dict[str, float] = {}

    if result.face_blendshapes:
        for category in result.face_blendshapes[0]:
            name = (
                category.category_name
                or category.display_name
                or ""
            ).strip().lower()

            if name:
                scores[f"blend_{name}"] = float(
                    category.score or 0.0
                )

    return {
        "head_roll_deg": head_roll,
        "frontal_offset": frontal_offset,
        "mouth_width": mouth_width,
        "lip_gap_ratio": lip_gap_ratio,
        "left_eye_aperture":
            left_eye_aperture,
        "right_eye_aperture":
            right_eye_aperture,
        "inner_brow_distance":
            inner_brow_distance,
        "left_cheek_to_mouth":
            left_cheek_to_mouth,
        "right_cheek_to_mouth":
            right_cheek_to_mouth,
        **scores,
    }


def _extract_valid_frames(
    frames: list[str],
) -> tuple[
    list[dict[str, float]],
    list[str],
]:
    valid: list[dict[str, float]] = []
    errors: list[str] = []

    for encoded in frames:
        try:
            valid.append(
                _extract_frame_metrics(encoded)
            )
        except TherapyQualityError as error:
            errors.append(str(error))

    return valid, errors


def _aggregate(
    metrics_list: list[
        dict[str, float]
    ],
) -> dict[str, float]:
    keys = set().union(
        *[
            metrics.keys()
            for metrics in metrics_list
        ]
    )

    return {
        key: float(
            median(
                metrics.get(key, 0.0)
                for metrics
                in metrics_list
            )
        )
        for key in keys
    }


def _blend(
    metrics: dict[str, float],
    name: str,
) -> float:
    return float(
        metrics.get(
            f"blend_{name.lower()}",
            0.0,
        )
    )


def _positive_delta(
    action: dict[str, float],
    neutral: dict[str, float],
    name: str,
) -> float:
    return max(
        0.0,
        _blend(action, name)
        - _blend(neutral, name),
    )


def _clamp01(value: float) -> float:
    return float(
        np.clip(value, 0.0, 1.0)
    )


def _eye_close_components(
    neutral: dict[str, float],
    action: dict[str, float],
) -> tuple[float, float]:
    left_blend = _positive_delta(
        action,
        neutral,
        "eyeBlinkLeft",
    )

    right_blend = _positive_delta(
        action,
        neutral,
        "eyeBlinkRight",
    )

    left_geometry = _clamp01(
        1.0
        - action["left_eye_aperture"]
        / (
            neutral["left_eye_aperture"]
            + EPSILON
        )
    )

    right_geometry = _clamp01(
        1.0
        - action["right_eye_aperture"]
        / (
            neutral["right_eye_aperture"]
            + EPSILON
        )
    )

    return (
        max(left_blend, left_geometry),
        max(right_blend, right_geometry),
    )


def _cheek_puff_detected(
    neutral: dict[str, float],
    action: dict[str, float],
) -> tuple[
    bool,
    dict[str, float],
]:
    threshold = THRESHOLDS["cheek_puff"]

    blend_change = _positive_delta(
        action,
        neutral,
        "cheekPuff",
    )

    blend_action = _blend(
        action,
        "cheekPuff",
    )

    mouth_width_decrease = max(
        0.0,
        1.0
        - action["mouth_width"]
        / (
            neutral["mouth_width"]
            + EPSILON
        ),
    )

    left_cheek_change = max(
        0.0,
        action["left_cheek_to_mouth"]
        - neutral["left_cheek_to_mouth"],
    )

    right_cheek_change = max(
        0.0,
        action["right_cheek_to_mouth"]
        - neutral["right_cheek_to_mouth"],
    )

    geometry_change = max(
        mouth_width_decrease,
        left_cheek_change,
        right_cheek_change,
    )

    lips_closed = (
        action["lip_gap_ratio"]
        <= threshold["max_lip_gap_ratio"]
    )

    detected = lips_closed and (
        blend_change
        >= threshold["min_blend_change"]
        or blend_action
        >= threshold["min_blend_action"]
        or geometry_change
        >= threshold["min_geometry_change"]
    )

    return detected, {
        "blend_change": blend_change,
        "blend_action": blend_action,
        "mouth_width_decrease":
            mouth_width_decrease,
        "left_cheek_change":
            left_cheek_change,
        "right_cheek_change":
            right_cheek_change,
        "geometry_change":
            geometry_change,
        "lip_gap_ratio":
            action["lip_gap_ratio"],
    }


def _pucker_detected(
    neutral: dict[str, float],
    action: dict[str, float],
) -> tuple[
    bool,
    dict[str, float],
]:
    threshold = THRESHOLDS["pucker"]

    pucker = _positive_delta(
        action,
        neutral,
        "mouthPucker",
    )

    funnel = _positive_delta(
        action,
        neutral,
        "mouthFunnel",
    )

    width_decrease = max(
        0.0,
        1.0
        - action["mouth_width"]
        / (
            neutral["mouth_width"]
            + EPSILON
        ),
    )

    detected = (
        pucker
        >= threshold["min_pucker_change"]
        and width_decrease
        >= threshold["min_width_decrease"]
    ) or (
        pucker
        >= threshold["alt_min_pucker_change"]
        and funnel
        >= threshold["alt_min_funnel_change"]
    )

    return detected, {
        "pucker_change": pucker,
        "funnel_change": funnel,
        "mouth_width_decrease":
            width_decrease,
    }


def _blowing_detected(
    neutral: dict[str, float],
    action: dict[str, float],
) -> tuple[
    bool,
    dict[str, float],
]:
    threshold = THRESHOLDS[
        "blowing_shape"
    ]

    pucker = _positive_delta(
        action,
        neutral,
        "mouthPucker",
    )

    funnel = _positive_delta(
        action,
        neutral,
        "mouthFunnel",
    )

    jaw_open = _blend(
        action,
        "jawOpen",
    )

    detected = (
        pucker
        >= threshold["min_pucker_change"]
        and funnel
        >= threshold["min_funnel_change"]
        and jaw_open
        <= threshold["max_jaw_open"]
    )

    return detected, {
        "pucker_change": pucker,
        "funnel_change": funnel,
        "jaw_open": jaw_open,
    }


def _lip_stretch_detected(
    neutral: dict[str, float],
    action: dict[str, float],
) -> tuple[
    bool,
    dict[str, float],
]:
    threshold = THRESHOLDS["lip_stretch"]

    left = _positive_delta(
        action,
        neutral,
        "mouthStretchLeft",
    )

    right = _positive_delta(
        action,
        neutral,
        "mouthStretchRight",
    )

    width_increase = max(
        0.0,
        action["mouth_width"]
        / (
            neutral["mouth_width"]
            + EPSILON
        )
        - 1.0,
    )

    strongest = max(left, right)

    detected = (
        strongest
        >= threshold["min_stretch_change"]
        and width_increase
        >= threshold["min_width_increase"]
    )

    return detected, {
        "left_stretch": left,
        "right_stretch": right,
        "mouth_width_increase":
            width_increase,
    }


def _eye_close_detected(
    neutral: dict[str, float],
    action: dict[str, float],
) -> tuple[
    bool,
    dict[str, float],
]:
    threshold = THRESHOLDS["eye_close"]

    left, right = _eye_close_components(
        neutral,
        action,
    )

    strongest = max(left, right)
    weakest = min(left, right)

    detected = (
        strongest
        >= threshold["min_strongest_closure"]
        and weakest
        >= threshold["min_weakest_closure"]
    )

    return detected, {
        "left_eye_closure": left,
        "right_eye_closure": right,
    }


def _brow_frown_detected(
    neutral: dict[str, float],
    action: dict[str, float],
) -> tuple[
    bool,
    dict[str, float],
]:
    threshold = THRESHOLDS["brow_frown"]

    left = _positive_delta(
        action,
        neutral,
        "browDownLeft",
    )

    right = _positive_delta(
        action,
        neutral,
        "browDownRight",
    )

    inner_decrease = max(
        0.0,
        1.0
        - action["inner_brow_distance"]
        / (
            neutral["inner_brow_distance"]
            + EPSILON
        ),
    )

    detected = (
        max(left, right)
        >= threshold["min_brow_down_change"]
        and inner_decrease
        >= threshold[
            "min_inner_brow_decrease"
        ]
    )

    return detected, {
        "left_brow_down": left,
        "right_brow_down": right,
        "inner_brow_decrease":
            inner_decrease,
    }


def _eye_close_lip_stretch_detected(
    neutral: dict[str, float],
    action: dict[str, float],
) -> tuple[
    bool,
    dict[str, Any],
]:
    eye_detected, eye_metrics = (
        _eye_close_detected(
            neutral,
            action,
        )
    )

    lip_detected, lip_metrics = (
        _lip_stretch_detected(
            neutral,
            action,
        )
    )

    return (
        eye_detected and lip_detected,
        {
            "eye": eye_metrics,
            "lip": lip_metrics,
        },
    )


def _smile_eye_close_detected(
    neutral: dict[str, float],
    action: dict[str, float],
) -> tuple[
    bool,
    dict[str, float],
]:
    threshold = THRESHOLDS[
        "smile_eye_close"
    ]

    smile_left = _positive_delta(
        action,
        neutral,
        "mouthSmileLeft",
    )

    smile_right = _positive_delta(
        action,
        neutral,
        "mouthSmileRight",
    )

    eye_left, eye_right = (
        _eye_close_components(
            neutral,
            action,
        )
    )

    detected = (
        max(smile_left, smile_right)
        >= threshold["min_smile_change"]
        and max(eye_left, eye_right)
        >= threshold["min_eye_closure"]
    )

    return detected, {
        "smile_left": smile_left,
        "smile_right": smile_right,
        "eye_left": eye_left,
        "eye_right": eye_right,
    }


def _max_true_streak(
    flags: list[bool],
) -> int:
    maximum = 0
    current = 0

    for flag in flags:
        if flag:
            current += 1
            maximum = max(
                maximum,
                current,
            )
        else:
            current = 0

    return maximum


def _classify_blink_sequence(
    neutral: dict[str, float],
    action_frames: list[
        dict[str, float]
    ],
) -> tuple[
    bool,
    dict[str, Any],
]:
    threshold = THRESHOLDS["blink"]

    closure_values: list[float] = []

    for action in action_frames:
        left, right = (
            _eye_close_components(
                neutral,
                action,
            )
        )

        closure_values.append(
            max(left, right)
        )

    closed_index: int | None = None

    for index, closure in enumerate(
        closure_values
    ):
        if (
            closure
            >= threshold["min_closed_value"]
        ):
            closed_index = index
            break

    reopened = False

    if closed_index is not None:
        for action in action_frames[
            closed_index + 1:
        ]:
            left_ratio = (
                action["left_eye_aperture"]
                / (
                    neutral[
                        "left_eye_aperture"
                    ]
                    + EPSILON
                )
            )

            right_ratio = (
                action["right_eye_aperture"]
                / (
                    neutral[
                        "right_eye_aperture"
                    ]
                    + EPSILON
                )
            )

            if (
                min(
                    left_ratio,
                    right_ratio,
                )
                >= threshold[
                    "min_reopen_ratio"
                ]
            ):
                reopened = True
                break

    return (
        closed_index is not None
        and reopened,
        {
            "closure_values":
                closure_values,
            "closed_frame_index":
                closed_index,
            "reopened_after_close":
                reopened,
        },
    )


def classify_therapy_realtime(
    exercise_key: str,
    neutral_frames: list[str],
    action_frames: list[str],
) -> dict[str, Any]:
    if exercise_key not in (
        SUPPORTED_MOVEMENTS
    ):
        raise ValueError(
            f"Gerakan tidak didukung: {exercise_key}"
        )

    neutral_valid, neutral_errors = (
        _extract_valid_frames(
            neutral_frames
        )
    )

    action_valid, action_errors = (
        _extract_valid_frames(
            action_frames
        )
    )

    if (
        len(neutral_valid)
        < MIN_VALID_NEUTRAL_FRAMES
    ):
        raise TherapyQualityError(
            neutral_errors[0]
            if neutral_errors
            else "Wajah netral tidak terbaca.",
            "neutral_not_detected",
        )

    valid_action_ratio = (
        len(action_valid)
        / max(
            len(action_frames),
            1,
        )
    )

    if (
        len(action_valid)
        < MIN_VALID_ACTION_FRAMES
        or valid_action_ratio
        < MIN_VALID_ACTION_RATIO
    ):
        raise TherapyQualityError(
            action_errors[0]
            if action_errors
            else (
                "Wajah belum terbaca "
                "dengan baik."
            ),
            "insufficient_valid_frames",
        )

    neutral = _aggregate(
        neutral_valid
    )

    if exercise_key == "blink":
        correct, blink_metrics = (
            _classify_blink_sequence(
                neutral,
                action_valid,
            )
        )

        return {
            "classification":
                "correct"
                if correct
                else "incorrect",
            "label":
                "Gerakan terapi benar."
                if correct
                else (
                    "Gerakan terapi belum benar. "
                    "Silakan coba kembali."
                ),
            "raw_metrics": {
                "valid_action_frames":
                    len(action_valid),
                "invalid_action_frames":
                    len(action_errors),
                "valid_action_ratio":
                    round(
                        valid_action_ratio,
                        4,
                    ),
                "blink": blink_metrics,
            },
            "algorithm_version":
                ALGORITHM_VERSION,
            "threshold_version":
                THRESHOLD_VERSION,
        }

    classifiers: dict[
        str,
        Callable[
            [
                dict[str, float],
                dict[str, float],
            ],
            tuple[
                bool,
                dict[str, Any],
            ],
        ],
    ] = {
        "cheek_puff":
            _cheek_puff_detected,
        "pucker":
            _pucker_detected,
        "blowing_shape":
            _blowing_detected,
        "lip_stretch":
            _lip_stretch_detected,
        "eye_close":
            _eye_close_detected,
        "brow_frown":
            _brow_frown_detected,
        "eye_close_lip_stretch":
            _eye_close_lip_stretch_detected,
        "smile_eye_close":
            _smile_eye_close_detected,
    }

    classifier = classifiers[
        exercise_key
    ]

    flags: list[bool] = []
    frame_metrics: list[
        dict[str, Any]
    ] = []

    for action in action_valid:
        detected, metrics = classifier(
            neutral,
            action,
        )

        flags.append(detected)
        frame_metrics.append(metrics)

    detected_frames = sum(
        1
        for flag in flags
        if flag
    )

    detected_ratio = (
        detected_frames
        / len(flags)
    )

    maximum_streak = _max_true_streak(
        flags
    )

    correct = (
        maximum_streak
        >= MIN_CONSECUTIVE_DETECTED_FRAMES
        or detected_ratio
        >= MIN_DETECTED_FRAME_RATIO
    )

    return {
        "classification":
            "correct"
            if correct
            else "incorrect",
        "label":
            "Gerakan terapi benar."
            if correct
            else (
                "Gerakan terapi belum benar. "
                "Silakan coba kembali."
            ),
        "raw_metrics": {
            "valid_action_frames":
                len(action_valid),
            "invalid_action_frames":
                len(action_errors),
            "valid_action_ratio":
                round(
                    valid_action_ratio,
                    4,
                ),
            "detected_frames":
                detected_frames,
            "detected_ratio":
                round(
                    detected_ratio,
                    4,
                ),
            "maximum_detected_streak":
                maximum_streak,
            "frame_flags": flags,
            "frame_metrics":
                frame_metrics,
        },
        "algorithm_version":
            ALGORITHM_VERSION,
        "threshold_version":
            THRESHOLD_VERSION,
    }
