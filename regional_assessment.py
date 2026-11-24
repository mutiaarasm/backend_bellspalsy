from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


REGIONAL_ASSESSMENT_VERSION = "regional-geometry-relaxed-gate-v2"


@dataclass(frozen=True)
class RegionalConfig:
    """
    Konfigurasi teknis untuk menerjemahkan fitur geometris menjadi
    kategori regional. Nilai ini bukan threshold resmi House-Brackmann.

    Setelah pengujian, nilai dapat dipindahkan ke file konfigurasi/DB
    tanpa mengubah rule House-Brackmann.
    """
    ai_normal_max: float = 0.20
    ai_clear_min: float = 0.35
    ai_severe_min: float = 0.55

    rest_mouth_angle_mild: float = 4.0
    rest_mouth_angle_clear: float = 8.0
    rest_mouth_angle_severe: float = 13.0

    forehead_no_movement_max: float = 0.004
    forehead_slight_max: float = 0.010
    forehead_good_min: float = 0.020
    forehead_ai_clear: float = 0.40
    # Selisih absolut minimum agar noise landmark tidak dianggap sebagai
    # kelemahan satu sisi. Nilai ini adalah toleransi teknis prototipe,
    # bukan threshold klinis.
    forehead_min_side_difference: float = 0.015

    eye_complete_min: float = 0.85
    eye_complete_reduced_min: float = 0.72
    eye_incomplete_min: float = 0.45
    eye_minimal_min: float = 0.15

    smile_reference: float = 0.025
    # snarl_reference: float = 0.020
    pucker_reference: float = 0.020
    pucker_width_reference: float = 0.10
    mouth_min_side_difference: float = 0.015


DEFAULT_CONFIG = RegionalConfig()


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _meaningful_asymmetry(
    feature: dict[str, Any],
    asymmetry_index: float,
    *,
    minimum_difference: float,
    minimum_index: float,
) -> bool:
    """Return true only when relative and absolute asymmetry agree."""
    left = abs(float(feature.get("left", 0.0)))
    right = abs(float(feature.get("right", 0.0)))
    absolute_difference = abs(left - right)

    return (
        asymmetry_index > minimum_index
        and absolute_difference >= minimum_difference
    )


def _side_from_votes(votes: list[str]) -> str:
    valid = [side for side in votes if side in {"left", "right"}]
    if len(valid) < 2:
        return "uncertain"

    left = valid.count("left")
    right = valid.count("right")

    if left == right:
        return "uncertain"
    return "left" if left > right else "right"


def assess_rest_region(
    feature: dict[str, Any],
    config: RegionalConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    eye_ai = float(feature["eye_ai"])
    brow_ai = float(feature["brow_ai"])
    mouth_angle = abs(float(feature["mouth_angle_deg"]))

    mild_indicators = sum([
        eye_ai > config.ai_normal_max,
        brow_ai > config.ai_normal_max,
        mouth_angle >= config.rest_mouth_angle_mild,
    ])

    clear_indicators = sum([
        eye_ai >= config.ai_clear_min,
        brow_ai >= config.ai_clear_min,
        mouth_angle >= config.rest_mouth_angle_clear,
    ])

    severe_indicators = sum([
        eye_ai >= config.ai_severe_min,
        brow_ai >= config.ai_severe_min,
        mouth_angle >= config.rest_mouth_angle_severe,
    ])

    if severe_indicators >= 2:
        category = "grossly_asymmetric"
        severity = 3
    elif clear_indicators >= 2:
        category = "asymmetric"
        severity = 2
    elif mild_indicators >= 2:
        category = "mild_asymmetry"
        severity = 1
    elif mild_indicators == 1:
        category = "grossly_symmetric"
        severity = 0
    else:
        category = "symmetric"
        severity = 0

    return {
        "category": category,
        "status": category,
        "severity": severity,
        "eye_asymmetry_index": round(eye_ai, 6),
        "brow_asymmetry_index": round(brow_ai, 6),
        "mouth_angle_deg": round(mouth_angle, 6),
        "indicator_count": {
            "mild": mild_indicators,
            "clear": clear_indicators,
            "severe": severe_indicators,
        },
    }


def assess_forehead_region(
    feature: dict[str, Any],
    config: RegionalConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    strength = float(feature["strength"])
    asymmetry = float(feature["asymmetry_index"])
    meaningful_asymmetry = _meaningful_asymmetry(
        feature,
        asymmetry,
        minimum_difference=config.forehead_min_side_difference,
        minimum_index=config.ai_normal_max,
    )

    if strength <= config.forehead_no_movement_max:
        category = "no_movement"
        severity = 3
    elif strength < config.forehead_slight_max:
        category = "slight_movement"
        severity = 3
    elif (
        strength < config.forehead_good_min
        or (
            meaningful_asymmetry
            and asymmetry >= config.forehead_ai_clear
        )
    ):
        category = "reduced_movement"
        severity = 2
    elif meaningful_asymmetry:
        category = "moderate_to_good"
        severity = 1
    else:
        category = "normal_movement"
        severity = 0

    return {
        "category": category,
        "status": category,
        "severity": severity,
        "strength": round(strength, 6),
        "asymmetry_index": round(asymmetry, 6),
        "meaningful_asymmetry": meaningful_asymmetry,
        "side_difference": round(
            abs(
                float(feature.get("left", 0.0))
                - float(feature.get("right", 0.0))
            ),
            6,
        ),
        "weaker_side": feature.get("weaker_side", "uncertain"),
    }


def assess_eye_region(
    feature: dict[str, Any],
    config: RegionalConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    minimum_closure = float(feature["minimum_closure"])
    mean_closure = float(feature["strength"])
    asymmetry = float(feature["asymmetry_index"])

    if (
        minimum_closure >= config.eye_complete_min
        and asymmetry <= config.ai_normal_max
    ):
        category = "complete_closure"
        severity = 0
    elif minimum_closure >= config.eye_complete_reduced_min:
        category = "complete_reduced"
        severity = 1
    elif minimum_closure >= config.eye_incomplete_min:
        category = "incomplete_closure"
        severity = 2
    elif minimum_closure >= config.eye_minimal_min:
        category = "minimal_closure"
        severity = 3
    else:
        category = "no_effective_closure"
        severity = 4

    return {
        "category": category,
        "status": category,
        "severity": severity,
        "minimum_closure": round(minimum_closure, 6),
        "mean_closure": round(mean_closure, 6),
        "asymmetry_index": round(asymmetry, 6),
        "weaker_side": feature.get("weaker_side", "uncertain"),
        "measurement_note": (
            "Sistem mengukur rasio penutupan, bukan usaha otot langsung."
        ),
    }


def assess_mouth_region(
    smile: dict[str, Any],
    # snarl: dict[str, Any],
    pucker: dict[str, Any],
    config: RegionalConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """
    Menilai fungsi mulut dari tiga gerakan:
    senyum, snarl, dan mencucu.

    Gerakan yang sangat kecil tetap dianggap data klinis.
    Fungsi ini tidak pernah meminta pengambilan ulang.
    """
    movements = {
        "senyum": smile,
        # "snarl": snarl,
        "mencucu": pucker,
    }

    asymmetry = {
        name: float(
            value.get(
                "asymmetry_index",
                0.0,
            )
        )
        for name, value in movements.items()
    }

    smile_function = _clip01(
        float(smile.get("strength", 0.0))
        / config.smile_reference
    )
    # snarl_function = _clip01(
    #     float(snarl.get("strength", 0.0))
    #     / config.snarl_reference
    # )
    pucker_function = max(
        _clip01(
            float(pucker.get("strength", 0.0))
            / config.pucker_reference
        ),
        _clip01(
            float(pucker.get("width_reduction", 0.0))
            / config.pucker_width_reference
        ),
    )

    component_function = {
        "senyum": smile_function,
        # "snarl": snarl_function,
        "mencucu": pucker_function,
    }

    movement_function = float(
        np.median(
            list(component_function.values())
        )
    )
    mean_asymmetry = float(
        np.median(
            list(asymmetry.values())
        )
    )

    component_categories: dict[str, str] = {}
    component_meaningful_asymmetry: dict[str, bool] = {}

    for name in movements:
        ai = asymmetry[name]
        function_value = component_function[name]
        meaningful_asymmetry = _meaningful_asymmetry(
            movements[name],
            ai,
            minimum_difference=config.mouth_min_side_difference,
            minimum_index=config.ai_normal_max,
        )
        component_meaningful_asymmetry[name] = meaningful_asymmetry

        if function_value < 0.08:
            component_categories[name] = "no_movement"
        elif function_value < 0.25:
            component_categories[name] = "barely_perceptible"
        elif meaningful_asymmetry and ai >= config.ai_severe_min:
            component_categories[name] = "severe_asymmetry"
        elif meaningful_asymmetry and ai >= config.ai_clear_min:
            component_categories[name] = "moderate_asymmetry"
        elif meaningful_asymmetry and ai > config.ai_normal_max:
            component_categories[name] = "mild_asymmetry"
        else:
            component_categories[name] = "symmetric"

    severity_map = {
        "symmetric": 0,
        "mild_asymmetry": 1,
        "moderate_asymmetry": 2,
        "severe_asymmetry": 3,
        "barely_perceptible": 4,
        "no_movement": 5,
    }

    severities = [
        severity_map[value]
        for value in component_categories.values()
    ]

    # Satu gerakan yang jelas asimetris tidak boleh hilang karena dua
    # gerakan lain kebetulan terlihat normal. Median sebelumnya membuat
    # kasus [asimetris, simetris, simetris] menjadi simetris.
    # Gunakan severity terburuk untuk mempertahankan sinyal klinis
    # per-gerakan; noise kecil sudah disaring oleh activity_floor pada
    # ekstraksi fitur.
    asymmetry_severities = [
        severity_map[component_categories[name]]
        for name in movements
        if component_meaningful_asymmetry[name]
    ]

    if asymmetry_severities:
        final_severity = max(asymmetry_severities)
    else:
        # Gerakan bilateral yang kecil/tidak tertangkap tidak otomatis
        # menjadi asimetri hanya karena severity no_movement tinggi.
        final_severity = int(round(float(np.median(severities))))

    category_by_severity = {
        0: "symmetric",
        1: "mild_asymmetry",
        2: "moderate_asymmetry",
        3: "severe_asymmetry",
        4: "barely_perceptible",
        5: "no_movement",
    }

    category = category_by_severity[
        final_severity
    ]

    side_votes = [
        str(
            movements[name].get(
                "weaker_side",
                "uncertain",
            )
        )
        for name in movements
        if component_categories[name]
        in {
            "moderate_asymmetry",
            "severe_asymmetry",
            "barely_perceptible",
            "no_movement",
        }
    ]

    weaker_side = _side_from_votes(
        side_votes
    )

    return {
        "category": category,
        "status": category,
        "severity": final_severity,
        "measurement_status": "valid",
        "function_score": round(
            movement_function,
            6,
        ),
        "mean_asymmetry_index": round(
            mean_asymmetry,
            6,
        ),
        "weaker_side": weaker_side,
        "weaker_side_votes": side_votes,
        "component_categories":
            component_categories,
        "component_meaningful_asymmetry":
            component_meaningful_asymmetry,
        "components": {
            "smile": round(
                smile_function,
                6,
            ),
            # "snarl": round(
            #     snarl_function,
            #     6,
            # ),
            "pucker": round(
                pucker_function,
                6,
            ),
        },
    }


def assess_regions(
    feature_values: dict[str, dict[str, Any]],
    config: RegionalConfig = DEFAULT_CONFIG,
) -> dict[str, dict[str, Any]]:
    """
    Mengubah fitur geometris menjadi kategori regional.

    Nilai gerakan kecil atau asimetri berat tetap diklasifikasikan
    dan tidak dianggap sebagai kegagalan teknis.
    """
    return {
        "rest": assess_rest_region(
            feature_values["rest"],
            config,
        ),
        "forehead": assess_forehead_region(
            feature_values["angkat_alis"],
            config,
        ),
        "eye": assess_eye_region(
            feature_values["tutup_mata"],
            config,
        ),
        "mouth": assess_mouth_region(
            feature_values["senyum"],
            # feature_values["snarl"],
            feature_values["mencucu"],
            config,
        ),
    }


REST_PROGRESS_SCORE = {
    "symmetric": 100.0,
    "grossly_symmetric": 90.0,
    "mild_asymmetry": 75.0,
    "asymmetric": 45.0,
    "grossly_asymmetric": 15.0,
}

FOREHEAD_PROGRESS_SCORE = {
    "normal_movement": 100.0,
    "moderate_to_good": 85.0,
    "reduced_movement": 60.0,
    "slight_movement": 30.0,
    "no_movement": 0.0,
}

EYE_PROGRESS_SCORE = {
    "complete_closure": 100.0,
    "complete_reduced": 85.0,
    "incomplete_closure": 55.0,
    "minimal_closure": 25.0,
    "no_effective_closure": 0.0,
}

MOUTH_PROGRESS_SCORE = {
    "symmetric": 100.0,
    "mild_asymmetry": 85.0,
    "moderate_asymmetry": 60.0,
    "severe_asymmetry": 35.0,
    "barely_perceptible": 15.0,
    "no_movement": 0.0,
}

LEVEL_PROGRESS_RANGE = {
    1: (90.0, 100.0),
    2: (75.0, 89.0),
    3: (55.0, 74.0),
    4: (35.0, 54.0),
    5: (15.0, 34.0),
    6: (0.0, 14.0),
}


def calculate_face_function_score(
    regional: dict[str, dict[str, Any]],
    level: int | None = None,
) -> float:
    """
    Menghasilkan skor perkembangan fungsi wajah 0-100.

    Skor berasal dari kategori regional yang sama dengan klasifikasi.
    Skor ini hanya untuk monitoring/progres, bukan penentu tingkat.
    """
    rest_category = str(regional["rest"]["category"])
    forehead_category = str(regional["forehead"]["category"])
    eye_category = str(regional["eye"]["category"])
    mouth_category = str(regional["mouth"]["category"])

    rest_score = REST_PROGRESS_SCORE.get(rest_category, 0.0)
    forehead_score = FOREHEAD_PROGRESS_SCORE.get(
        forehead_category,
        0.0,
    )
    eye_score = EYE_PROGRESS_SCORE.get(eye_category, 0.0)
    mouth_score = MOUTH_PROGRESS_SCORE.get(mouth_category, 0.0)

    raw_score = (
        rest_score * 0.15
        + forehead_score * 0.20
        + eye_score * 0.30
        + mouth_score * 0.35
    )

    # Menjaga persentase tetap konsisten dengan Tingkat I-VI.
    if level in LEVEL_PROGRESS_RANGE:
        minimum, maximum = LEVEL_PROGRESS_RANGE[level]
        raw_score = max(minimum, min(raw_score, maximum))

    return round(float(np.clip(raw_score, 0.0, 100.0)), 2)

def infer_affected_side(
    regional: dict[str, dict[str, Any]],
) -> str:
    all_normal = (
        regional["rest"].get("severity", 0) == 0
        and regional["forehead"].get("severity", 0) == 0
        and regional["eye"].get("severity", 0) == 0
        and regional["mouth"].get("severity", 0) == 0
    )

    if all_normal:
        return "symmetric"

    votes = [
        regional["forehead"].get("weaker_side"),
        regional["eye"].get("weaker_side"),
        regional["mouth"].get("weaker_side"),
    ]

    side = _side_from_votes([
        str(value)
        for value in votes
    ])

    if side == "left":
        return "left_image"

    if side == "right":
        return "right_image"

    return "uncertain"
