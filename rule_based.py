from collections import Counter
from statistics import mean


ALGORITHM_VERSION = "geometry-rule-v2.1"
THRESHOLD_VERSION = "pilot-v1"

# Threshold awal/pilot.
# Belum boleh dianggap threshold klinis final.
THRESHOLDS = {
    "rest_eye_delta": 0.0078,
    "rest_eye_ai": 0.12,
    "rest_brow_ai": 0.15,
    "rest_mouth_angle_deg": 5.85,

    "brow_asymmetry_index": 0.25,
    "brow_angle_change_deg": 3.35,

    "eye_closure_asymmetry": 0.20,
    "minimum_eye_closure": 0.70,

    "smile_asymmetry_index": 0.25,
    "smile_angle_change_deg": 5.85,

    "snarl_asymmetry_index": 0.25,
    "pucker_asymmetry_index": 0.25,
}


def _clip01(value):
    return max(0.0, min(1.0, float(value)))


def _threshold_ratio(value, threshold):
    if threshold <= 0:
        return 0.0

    return _clip01(
        float(value) / float(threshold)
    )


def _dominant_side(side_votes):
    valid_votes = [
        side
        for side in side_votes
        if side in {"left", "right"}
    ]

    if len(valid_votes) < 2:
        return "uncertain"

    counts = Counter(valid_votes)
    side, count = counts.most_common(1)[0]

    if count < 2:
        return "uncertain"

    return side


def _get_feedback(total_score, abnormal_areas):
    if total_score == 0:
        return {
            "title":
                "Tidak ditemukan asimetri pada wajah.",
            "recommendation":
                "Gerakan wajah yang dianalisis berada dalam "
                "rentang aturan sistem. Hasil ini bukan diagnosis medis.",
        }

    area_text = ", ".join(
        abnormal_areas
    ) if abnormal_areas else "beberapa area wajah"

    if total_score <= 2:
        return {
            "title":
                "Terdeteksi asimetri wajah ringan.",
            "recommendation":
                f"Asimetri terdeteksi pada {area_text}. "
                "Ulangi pengukuran bila kualitas pengambilan meragukan "
                "dan gunakan monitoring untuk melihat perubahan.",
        }

    return {
        "title":
            "Terdeteksi asimetri pada beberapa gerakan.",
        "recommendation":
            f"Asimetri terdeteksi pada {area_text}. "
            "Hasil merupakan skrining visual, bukan diagnosis Bell's Palsy. "
            "Pertimbangkan pemeriksaan oleh tenaga kesehatan.",
    }


def classify_result(analysis):
    if not isinstance(analysis, dict):
        raise ValueError(
            "analysis harus berupa dictionary"
        )

    quality_status = analysis.get(
        "quality_status"
    )

    if quality_status != "valid":
        return {
            "quality_status": quality_status,
            "classification":
                "Tidak dapat dinilai",
            "screening_result":
                "Pengukuran tidak memenuhi kualitas minimum",
            "total_score": None,
            "total_skor": None,
            "percentage": None,
            "asymmetry_percentage": None,
            "scores": {},
            "movement_scores": {},
            "abnormal_areas": [],
            "affected_side": "uncertain",
            "features":
                analysis.get("feature_values", {}),
            "feature_values":
                analysis.get("feature_values", {}),
            "valid_frame_count":
                analysis.get("valid_frame_count", {}),
            "rejected_frame_count":
                analysis.get("rejected_frame_count", {}),
            "algorithm_version":
                ALGORITHM_VERSION,
            "threshold_version":
                THRESHOLD_VERSION,
            "feedback": {
                "title":
                    "Pengukuran tidak dapat dinilai.",
                "recommendation":
                    "Ulangi deteksi dengan posisi wajah lurus, "
                    "pencahayaan cukup, dan ikuti seluruh gerakan.",
            },
        }

    features = analysis.get(
        "feature_values"
    )

    if not isinstance(features, dict):
        raise ValueError(
            "feature_values tidak ditemukan"
        )

    required_keys = {
        "rest",
        "angkat_alis",
        "tutup_mata",
        "senyum",
        "snarl",
        "mencucu",
    }

    missing_keys = (
        required_keys - set(features.keys())
    )

    if missing_keys:
        raise ValueError(
            f"Fitur gerakan belum lengkap: "
            f"{sorted(missing_keys)}"
        )

    rest = features["rest"]
    brow = features["angkat_alis"]
    eye = features["tutup_mata"]
    smile = features["senyum"]
    snarl = features["snarl"]
    pucker = features["mencucu"]

    # =========================
    # RULE POSISI ISTIRAHAT
    # =========================

    rest_eye_abnormal = (
        rest["eye_delta"]
        > THRESHOLDS["rest_eye_delta"]
        or
        rest["eye_ai"]
        > THRESHOLDS["rest_eye_ai"]
    )

    rest_brow_abnormal = (
        rest["brow_ai"]
        > THRESHOLDS["rest_brow_ai"]
    )

    rest_mouth_abnormal = (
        rest["mouth_angle_deg"]
        > THRESHOLDS[
            "rest_mouth_angle_deg"
        ]
    )

    rest_abnormal = (
        rest_eye_abnormal
        or rest_brow_abnormal
        or rest_mouth_abnormal
    )

    # =========================
    # RULE GERAKAN AKTIF
    # =========================

    brow_abnormal = (
        brow["asymmetry_index"]
        > THRESHOLDS[
            "brow_asymmetry_index"
        ]
        or
        brow["angle_change_deg"]
        > THRESHOLDS[
            "brow_angle_change_deg"
        ]
    )

    eye_abnormal = (
        eye["asymmetry_index"]
        > THRESHOLDS[
            "eye_closure_asymmetry"
        ]
        or
        eye["minimum_closure"]
        < THRESHOLDS[
            "minimum_eye_closure"
        ]
    )

    smile_abnormal = (
        smile["asymmetry_index"]
        > THRESHOLDS[
            "smile_asymmetry_index"
        ]
        or
        smile["angle_change_deg"]
        > THRESHOLDS[
            "smile_angle_change_deg"
        ]
    )

    snarl_abnormal = (
        snarl["asymmetry_index"]
        > THRESHOLDS[
            "snarl_asymmetry_index"
        ]
    )

    pucker_abnormal = (
        pucker["asymmetry_index"]
        > THRESHOLDS[
            "pucker_asymmetry_index"
        ]
    )

    movement_scores = {
        "diam": int(rest_abnormal),
        "angkat_alis": int(brow_abnormal),
        "tutup_mata": int(eye_abnormal),
        "senyum": int(smile_abnormal),
        "snarl": int(snarl_abnormal),
        "mencucu": int(pucker_abnormal),
    }

    total_score = sum(
        movement_scores.values()
    )

    # =========================
    # AREA ABNORMAL
    # =========================

    abnormal_areas = set()

    if rest_eye_abnormal or eye_abnormal:
        abnormal_areas.add("Mata")

    if rest_brow_abnormal or brow_abnormal:
        abnormal_areas.add("Alis")

    if (
        rest_mouth_abnormal
        or smile_abnormal
        or pucker_abnormal
    ):
        abnormal_areas.add("Mulut")

    if snarl_abnormal:
        abnormal_areas.add("Bibir Atas")

    abnormal_areas = sorted(
        abnormal_areas
    )

    # =========================
    # KONSISTENSI SISI
    # =========================

    side_votes = []

    for movement, abnormal in (
        (brow, brow_abnormal),
        (eye, eye_abnormal),
        (smile, smile_abnormal),
        (snarl, snarl_abnormal),
        (pucker, pucker_abnormal),
    ):
        if abnormal:
            side_votes.append(
                movement.get(
                    "weaker_side",
                    "uncertain",
                )
            )

    affected_side = _dominant_side(
        side_votes
    )

    # =========================
    # KATEGORI OPERASIONAL
    # =========================

    if total_score == 0:
        classification = (
            "Tidak terdeteksi asimetri "
            "wajah bermakna"
        )

    elif total_score <= 2:
        classification = (
            "Asimetri wajah ringan"
        )

    else:
        classification = (
            "Asimetri pada beberapa gerakan"
        )

    # =========================
    # INDEKS KONTINU
    # =========================

    rest_component = max(
        _clip01(rest["eye_ai"]),
        _clip01(rest["brow_ai"]),
        _threshold_ratio(
            rest["mouth_angle_deg"],
            THRESHOLDS[
                "rest_mouth_angle_deg"
            ],
        ),
    )

    brow_component = max(
        _clip01(
            brow["asymmetry_index"]
        ),
        _threshold_ratio(
            brow["angle_change_deg"],
            THRESHOLDS[
                "brow_angle_change_deg"
            ],
        ),
    )

    eye_component = max(
        _clip01(
            eye["asymmetry_index"]
        ),
        _clip01(
            1.0 - eye["minimum_closure"]
        ),
    )

    smile_component = max(
        _clip01(
            smile["asymmetry_index"]
        ),
        _threshold_ratio(
            smile["angle_change_deg"],
            THRESHOLDS[
                "smile_angle_change_deg"
            ],
        ),
    )

    snarl_component = _clip01(
        snarl["asymmetry_index"]
    )

    pucker_component = _clip01(
        pucker["asymmetry_index"]
    )

    composite_asymmetry_index = float(
        mean([
            rest_component,
            brow_component,
            eye_component,
            smile_component,
            snarl_component,
            pucker_component,
        ])
    )

    asymmetry_percentage = round(
        composite_asymmetry_index * 100.0,
        2,
    )

    # Semakin tinggi percentage,
    # semakin simetris/baik.
    percentage = round(
        (
            1.0
            - composite_asymmetry_index
        ) * 100.0,
        2,
    )

    feedback = _get_feedback(
        total_score,
        abnormal_areas,
    )

    return {
        "quality_status": "valid",

        "classification": classification,
        "screening_result": classification,

        "total_score": total_score,

        # Kompatibilitas kode Flutter lama.
        "total_skor": total_score,

        "percentage": percentage,
        "asymmetry_percentage":
            asymmetry_percentage,

        "composite_asymmetry_index":
            composite_asymmetry_index,

        "movement_scores":
            movement_scores,

        # Kompatibilitas ResultView lama.
        "scores": movement_scores,

        "resting_scores": {
            "mata": int(
                rest_eye_abnormal
            ),
            "alis": int(
                rest_brow_abnormal
            ),
            "mulut": int(
                rest_mouth_abnormal
            ),
        },

        "abnormal_areas":
            abnormal_areas,

        "affected_side":
            affected_side,

        "feature_values": features,

        # Kompatibilitas lama.
        "features": features,

        "valid_frame_count":
            analysis.get(
                "valid_frame_count",
                {},
            ),

        "rejected_frame_count":
            analysis.get(
                "rejected_frame_count",
                {},
            ),

        "algorithm_version":
            ALGORITHM_VERSION,

        "threshold_version":
            THRESHOLD_VERSION,

        "feedback": feedback,

        "disclaimer": (
            "Hasil merupakan skrining "
            "asimetri wajah berbasis fitur "
            "geometris dan bukan diagnosis "
            "Bell's Palsy."
        ),
    }