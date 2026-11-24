from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required

from detection_service import REQUIRED_MOVEMENTS, analyze_detection_frames
from models import DetectionHistory, db


detection_hb_bp = Blueprint("detection_hb", __name__)
MAX_MONITORING_SESSIONS = 7


def _current_user_id() -> int:
    identity = get_jwt_identity()
    try:
        return int(identity)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid_user_identity") from error


def _extract_frames(payload: dict[str, Any]) -> dict[str, Any]:
    frames = payload.get("frames")
    if isinstance(frames, dict):
        return frames
    return {movement: payload.get(movement) for movement in REQUIRED_MOVEMENTS}


def _next_monitoring_session_number(user_id: int) -> int:
    return DetectionHistory.query.filter_by(user_id=user_id, mode="monitoring").count() + 1


def _save_history(*, user_id: int, mode: str, session_number: int | None, result: dict[str, Any]) -> DetectionHistory:
    history = DetectionHistory(
        user_id=user_id,
        mode=mode,
        session_number=session_number,
        hb_grade=result.get("hb_grade"),
        hb_roman=result.get("hb_roman"),
        hb_label=result.get("hb_label"),
        classification_status=result.get("classification_status", "classified"),
        face_function_score=result.get("face_function_score"),
        affected_side=result.get("affected_side", "uncertain"),
        regional_result=result.get("regional_result", {}),
        feature_values=result.get("feature_values", {}),
        algorithm_version=result.get("algorithm_version", "unknown"),
        threshold_version=result.get("threshold_version", "unknown"),
    )
    db.session.add(history)
    db.session.commit()
    return history


@detection_hb_bp.post("/api/detection/analyze")
@jwt_required()
def analyze_detection():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(success=False, quality_status="invalid_request_body", message="Body request harus berupa JSON."), 400

    mode = str(payload.get("mode", "quick")).strip().lower()
    if mode not in {"quick", "monitoring"}:
        return jsonify(success=False, quality_status="invalid_mode", message="Mode hanya boleh quick atau monitoring."), 400

    try:
        user_id = _current_user_id()
    except ValueError:
        return jsonify(success=False, message="Identitas pengguna tidak valid."), 401

    session_number: int | None = None
    if mode == "monitoring":
        session_number = _next_monitoring_session_number(user_id)
        if session_number > MAX_MONITORING_SESSIONS:
            return jsonify(success=False, quality_status="monitoring_complete", message="Tujuh sesi monitoring sudah selesai. Reset program sebelum memulai ulang."), 409

    result = analyze_detection_frames(_extract_frames(payload))
    if result.get("quality_status") != "valid":
        return jsonify(success=False, **result), 422

    # Do not save an uncertain result as a clinical-looking grade.
    if result.get("classification_status") != "classified":
        return jsonify(success=False, **result), 422

    try:
        history = _save_history(user_id=user_id, mode=mode, session_number=session_number, result=result)
    except Exception as error:
        db.session.rollback()
        print("SAVE DETECTION HISTORY ERROR:", error)
        return jsonify(success=False, message="Hasil terukur tetapi gagal disimpan."), 500

    return jsonify(success=True, **result, history_id=history.id, mode=mode, session_number=session_number), 200


@detection_hb_bp.get("/api/detection/history")
@jwt_required()
def get_detection_history():
    user_id = _current_user_id()
    histories = DetectionHistory.query.filter_by(user_id=user_id).order_by(DetectionHistory.created_at.desc()).all()
    return jsonify(success=True, data=[h.to_dict(include_features=False) for h in histories]), 200


@detection_hb_bp.get("/api/detection/history/<int:history_id>")
@jwt_required()
def get_detection_history_detail(history_id: int):
    user_id = _current_user_id()
    history = DetectionHistory.query.filter_by(id=history_id, user_id=user_id).first()
    if history is None:
        return jsonify(success=False, message="Riwayat deteksi tidak ditemukan."), 404
    return jsonify(success=True, data=history.to_dict(include_features=True)), 200


@detection_hb_bp.get("/api/monitoring/history")
@jwt_required()
def get_monitoring_history():
    user_id = _current_user_id()
    histories = (DetectionHistory.query.filter_by(user_id=user_id, mode="monitoring")
                 .order_by(DetectionHistory.session_number.asc(), DetectionHistory.created_at.asc())
                 .limit(MAX_MONITORING_SESSIONS).all())
    return jsonify(success=True, data=[h.to_dict(include_features=False) for h in histories]), 200


@detection_hb_bp.delete("/api/monitoring/reset")
@jwt_required()
def reset_monitoring_history():
    user_id = _current_user_id()
    try:
        deleted_count = DetectionHistory.query.filter_by(user_id=user_id, mode="monitoring").delete(synchronize_session=False)
        db.session.commit()
    except Exception as error:
        db.session.rollback()
        print("RESET MONITORING ERROR:", error)
        return jsonify(success=False, message="Program monitoring gagal direset."), 500
    return jsonify(success=True, deleted_count=deleted_count, message="Program monitoring berhasil direset."), 200
