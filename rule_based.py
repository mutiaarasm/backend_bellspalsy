T_MATA = 0.0078
T_ALIS = 3.35
T_MULUT = 5.85


def movement_score(movement, f_mata, f_alis, f_mulut):

    # Diam cek semua area
    if movement == "diam":
        return int(
            f_mata > T_MATA or
            f_alis > T_ALIS or
            f_mulut > T_MULUT
        )

    # ANGKAT ALIS → fokus alis
    if movement == "angkat_alis":
        return int(f_alis > T_ALIS)

    # TUTUP MATA → fokus mata
    if movement == "tutup_mata":
        return int(f_mata > T_MATA)

    # SENYUM → fokus mulut
    if movement == "senyum":
        return int(f_mulut > T_MULUT)

    # MENCUCU → fokus mulut
    if movement == "mencucu":
        return int(f_mulut > T_MULUT)

    return 0


def classify_total_score(total_score):

    if total_score == 0:
        return "Normal"

    if total_score <= 2:
        return "Asimetri Ringan"

    return "Terdeteksi indikasi Bell's Palsy"


def get_feedback(total_score, abnormal_areas):

    if total_score == 0:
        return {
            "title": "Wajah Anda terdeteksi simetris.",
            "recommendation":
                "Tidak ditemukan indikasi asimetri saraf wajah. "
                "Tetap jaga kesehatan dan istirahat yang cukup."
        }

    if total_score <= 2:
        area_text = ", ".join(abnormal_areas)

        return {
            "title": "Terdeteksi asimetri ringan.",
            "recommendation":
                f"Sistem mendeteksi asimetri ringan pada area {area_text}. "
                "Gunakan fitur monitoring untuk memantau perubahan "
                "dan lakukan terapi exercise secara berkala."
        }

    return {
        "title": "Terdeteksi indikasi Bell's Palsy.",
        "recommendation":
            "Sistem mendeteksi pola asimetri wajah yang konsisten "
            "pada beberapa gerakan wajah. "
            "Disarankan memulai sesi monitoring dan terapi awal. "
            "Jika gejala muncul mendadak atau disertai keluhan berat, "
            "segera konsultasikan ke tenaga medis."
    }


def classify_result(movement_features):

    scores = {}
    abnormal_areas = set()

    for movement, features in movement_features.items():

        f_mata = features.get("mata", 0)
        f_alis = features.get("alis", 0)
        f_mulut = features.get("mulut", 0)

        # =========================
        # SCORING PER GERAKAN
        # =========================

        score = movement_score(
            movement,
            f_mata,
            f_alis,
            f_mulut
        )

        scores[movement] = score

        # =========================
        # ABNORMAL AREA
        # hanya area relevan
        # =========================

        # DIAM
        if movement == "diam":

            if f_mata > T_MATA:
                abnormal_areas.add("Mata")

            if f_alis > T_ALIS:
                abnormal_areas.add("Alis")

            if f_mulut > T_MULUT:
                abnormal_areas.add("Mulut")

        # ANGKAT ALIS
        elif movement == "angkat_alis":

            if f_alis > T_ALIS:
                abnormal_areas.add("Alis")

        # TUTUP MATA
        elif movement == "tutup_mata":

            if f_mata > T_MATA:
                abnormal_areas.add("Mata")

        # SENYUM
        elif movement == "senyum":

            if f_mulut > T_MULUT:
                abnormal_areas.add("Mulut")

        # MENCUCU
        elif movement == "mencucu":

            if f_mulut > T_MULUT:
                abnormal_areas.add("Mulut")

    # =========================
    # TOTAL SCORE
    # =========================

    total_score = sum(scores.values())

    # =========================
    # CLASSIFICATION
    # =========================

    classification = classify_total_score(total_score)

    # =========================
    # FEEDBACK
    # =========================

    feedback = get_feedback(
        total_score,
        list(abnormal_areas)
    )

    percentage = round((total_score / 5) * 100)

    return {
        "percentage": percentage,
        "total_skor": total_score,
        "classification": classification,
        "scores": scores,
        "features": movement_features,
        "abnormal_areas": list(abnormal_areas),
        "feedback": feedback
    }