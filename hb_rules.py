from __future__ import annotations

from typing import Any

HB_RULE_VERSION = "hb-characteristic-rules-relaxed-gate-v2"

HB_LABELS = {
    1: "Fungsi wajah normal",
    2: "Disfungsi ringan",
    3: "Disfungsi sedang",
    4: "Disfungsi sedang-berat",
    5: "Disfungsi berat",
    6: "Paralisis total",
}

HB_ROMANS = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
}


def _result(
    *,
    grade: int | None,
    status: str,
    rule: str,
    message: str,
    inconsistent_regions: list[str] | None = None,
) -> dict[str, Any]:
    if grade is None:
        return {
            "status": status,
            "grade": None,
            "roman": None,
            "label": "Pengukuran perlu diulang",
            "matched_rule": rule,
            "message": message,
            "inconsistent_regions": inconsistent_regions or [],
            "rule_version": HB_RULE_VERSION,
        }

    return {
        "status": status,
        "grade": grade,
        "roman": HB_ROMANS[grade],
        "label": HB_LABELS[grade],
        "matched_rule": rule,
        "message": message,
        "inconsistent_regions": inconsistent_regions or [],
        "rule_version": HB_RULE_VERSION,
    }


def validate_regional_consistency(
    regional: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Menolak pola yang tidak konsisten sebelum grading.

    Contoh:
    - Kondisi diam tampak normal, tetapi semua gerakan dinilai tidak ada.
    - Sisi lemah berpindah-pindah tanpa dukungan mayoritas.
    - Hanya satu indikator mulut yang abnormal.
    """
    inconsistent: list[str] = []

    rest = regional["rest"]
    forehead = regional["forehead"]
    eye = regional["eye"]
    mouth = regional["mouth"]

    if (
        rest["category"] == "symmetric"
        and forehead["category"] == "no_movement"
        and eye["category"] == "no_effective_closure"
        and mouth["category"] == "no_movement"
    ):
        inconsistent.append("global_pattern")

    return {
        "is_consistent": not inconsistent,
        "inconsistent_regions": inconsistent,
    }


def classify_house_brackmann(
    regional: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Menentukan Tingkat I-VI hanya dari kategori regional.

    Satu konsep:
    fitur geometris -> kategori regional -> Tingkat I-VI.

    Nilai persentase progres tidak digunakan untuk menentukan tingkat.
    """
    consistency = validate_regional_consistency(regional)

    if not consistency["is_consistent"]:
        return _result(
            grade=None,
            status="repeat_required",
            rule="regional_inconsistency",
            message=(
                "Hasil antarbagian wajah belum konsisten. "
                "Silakan ulangi pengambilan."
            ),
            inconsistent_regions=consistency["inconsistent_regions"],
        )

    rest = regional["rest"]["category"]
    forehead = regional["forehead"]["category"]
    eye = regional["eye"]["category"]
    mouth = regional["mouth"]["category"]

    # Tingkat VI — paralisis total.
    if (
        rest == "grossly_asymmetric"
        and forehead == "no_movement"
        and eye == "no_effective_closure"
        and mouth == "no_movement"
    ):
        return _result(
            grade=6,
            status="classified",
            rule="level_vi_total_paralysis",
            message=(
                "Tidak ditemukan gerakan efektif pada dahi, "
                "mata, dan mulut."
            ),
        )

    # Tingkat V — fungsi sangat sedikit.
    if (
        rest in {"asymmetric", "grossly_asymmetric"}
        and forehead in {"slight_movement", "no_movement"}
        and eye in {"minimal_closure", "no_effective_closure"}
        and mouth in {"barely_perceptible", "no_movement"}
    ):
        return _result(
            grade=5,
            status="classified",
            rule="level_v_severe_dysfunction",
            message=(
                "Gerakan wajah sangat terbatas dan "
                "penutupan mata tidak efektif."
            ),
        )

    # Tingkat IV — mata tidak menutup lengkap disertai
    # penurunan fungsi dahi dan mulut.
    if (
        forehead in {
            "reduced_movement",
            "slight_movement",
            "no_movement",
        }
        and eye == "incomplete_closure"
        and mouth in {
            "moderate_asymmetry",
            "severe_asymmetry",
            "barely_perceptible",
        }
    ):
        return _result(
            grade=4,
            status="classified",
            rule="level_iv_moderately_severe",
            message=(
                "Gerakan dahi dan mulut berkurang jelas, "
                "serta mata belum dapat menutup lengkap."
            ),
        )

    # Tingkat III — mata masih menutup, tetapi fungsi mulut
    # mengalami penurunan sedang.
    if (
        rest in {
            "symmetric",
            "grossly_symmetric",
            "mild_asymmetry",
        }
        and forehead in {
            "moderate_to_good",
            "reduced_movement",
        }
        and eye in {
            "complete_closure",
            "complete_reduced",
        }
        and mouth == "moderate_asymmetry"
    ):
        return _result(
            grade=3,
            status="classified",
            rule="level_iii_moderate_dysfunction",
            message=(
                "Mata masih dapat menutup, tetapi gerakan "
                "dahi dan mulut mengalami penurunan sedang."
            ),
        )

    # Tingkat I — seluruh bagian harus normal.
    if (
        rest == "symmetric"
        and forehead == "normal_movement"
        and eye == "complete_closure"
        and mouth == "symmetric"
    ):
        return _result(
            grade=1,
            status="classified",
            rule="level_i_normal_function",
            message="Fungsi wajah berada dalam rentang normal.",
        )

    # Tingkat II — terdapat minimal satu penurunan ringan,
    # tetapi mata masih dapat menutup lengkap.
    if (
        rest in {
            "symmetric",
            "grossly_symmetric",
            "mild_asymmetry",
        }
        and forehead in {
            "normal_movement",
            "moderate_to_good",
            "reduced_movement",
        }
        and eye in {
            "complete_closure",
            "complete_reduced",
        }
        and mouth in {
            "symmetric",
            "mild_asymmetry",
        }
        and (
            rest != "symmetric"
            or forehead != "normal_movement"
            or eye != "complete_closure"
            or mouth != "symmetric"
        )
    ):
        return _result(
            grade=2,
            status="classified",
            rule="level_ii_mild_dysfunction",
            message=(
                "Terdapat penurunan fungsi ringan, tetapi "
                "mata masih dapat menutup lengkap."
            ),
        )

    # Pola regional yang tidak persis cocok tidak dianggap
    # sebagai kegagalan teknis. Sistem memilih tingkat konservatif
    # berdasarkan tingkat keparahan regional tertinggi.
    severities = [
        int(regional[name].get("severity", 0))
        for name in (
            "rest",
            "forehead",
            "eye",
            "mouth",
        )
    ]

    maximum_severity = max(severities)

    if maximum_severity <= 1:
        fallback_grade = 2
        fallback_rule = "level_ii_regional_fallback"
        fallback_message = (
            "Terdapat penurunan fungsi ringan pada satu "
            "atau beberapa bagian wajah."
        )
    elif maximum_severity == 2:
        fallback_grade = 3
        fallback_rule = "level_iii_regional_fallback"
        fallback_message = (
            "Terdapat penurunan fungsi sedang pada satu "
            "atau beberapa bagian wajah."
        )
    elif maximum_severity == 3:
        fallback_grade = 4
        fallback_rule = "level_iv_regional_fallback"
        fallback_message = (
            "Terdapat penurunan fungsi sedang-berat pada "
            "satu atau beberapa bagian wajah."
        )
    elif maximum_severity == 4:
        fallback_grade = 5
        fallback_rule = "level_v_regional_fallback"
        fallback_message = (
            "Gerakan wajah sangat terbatas."
        )
    else:
        fallback_grade = 6
        fallback_rule = "level_vi_regional_fallback"
        fallback_message = (
            "Tidak ditemukan gerakan efektif pada beberapa "
            "bagian wajah."
        )

    return _result(
        grade=fallback_grade,
        status="classified",
        rule=fallback_rule,
        message=fallback_message,
    )
