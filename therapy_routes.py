from __future__ import annotations

import traceback
from typing import Any

from flask import Blueprint, jsonify, request
from flask_jwt_extended import (
    get_jwt_identity,
    jwt_required,
)

from models import TherapyResult, db
from therapy_service import (
    SUPPORTED_MOVEMENTS,
    TherapyQualityError,
    classify_therapy_realtime,
)


therapy_bp = Blueprint(
    "therapy",
    __name__,
)


def _frame_list(
    data: dict[str, Any],
    key: str,
) -> list[str] | None:
    value = data.get(key)

    if not isinstance(value, list):
        return None

    return [
        item
        for item in value
        if isinstance(item, str)
        and item.strip()
    ]


@therapy_bp.post(
    "/api/therapy/classify-realtime"
)
@jwt_required()
def classify_therapy_realtime_route():
    try:
        data = request.get_json(
            silent=True,
        ) or {}

        exercise_key = (
            data.get("exercise_key") or ""
        ).strip()

        video_id = (
            data.get("video_id") or ""
        ).strip()

        day = data.get("day")

        neutral_frames = _frame_list(
            data,
            "neutral_frames",
        )

        action_frames = _frame_list(
            data,
            "action_frames",
        )

        if exercise_key not in SUPPORTED_MOVEMENTS:
            return jsonify({
                "success": False,
                "message":
                    "Jenis gerakan tidak didukung.",
            }), 400

        try:
            day = int(day)
        except (TypeError, ValueError):
            return jsonify({
                "success": False,
                "message":
                    "Hari terapi tidak valid.",
            }), 400

        if day < 1 or day > 7:
            return jsonify({
                "success": False,
                "message":
                    "Hari terapi harus 1 sampai 7.",
            }), 400

        if not neutral_frames:
            return jsonify({
                "success": False,
                "message":
                    "Frame netral tidak tersedia.",
            }), 400

        if (
            not action_frames
            or len(action_frames) < 5
        ):
            return jsonify({
                "success": False,
                "message":
                    "Minimal lima frame gerakan diperlukan.",
            }), 400

        result = classify_therapy_realtime(
            exercise_key=exercise_key,
            neutral_frames=neutral_frames,
            action_frames=action_frames,
        )

        user_id = int(
            get_jwt_identity()
        )

        therapy_result = TherapyResult(
            user_id=user_id,
            day=day,
            exercise_key=exercise_key,
            video_id=video_id or None,
            classification=result[
                "classification"
            ],
            label=result["label"],
            raw_metrics=result.get(
                "raw_metrics"
            ),
            algorithm_version=result.get(
                "algorithm_version"
            ),
            threshold_version=result.get(
                "threshold_version"
            ),
        )

        db.session.add(therapy_result)
        db.session.commit()

        return jsonify({
            "success": True,
            "result_id": therapy_result.id,
            "day": day,
            "exercise_key": exercise_key,
            "video_id": video_id,
            "classification":
                result["classification"],
            "label": result["label"],
        }), 200

    except TherapyQualityError as error:
        return jsonify({
            "success": False,
            "quality_status":
                error.quality_status,
            "classification":
                "invalid_capture",
            "label": str(error),
            "message": str(error),
        }), 422

    except Exception as error:
        db.session.rollback()
        traceback.print_exc()

        return jsonify({
            "success": False,
            "message":
                "Terjadi kesalahan saat "
                "memeriksa gerakan terapi.",
            "error": str(error),
        }), 500
