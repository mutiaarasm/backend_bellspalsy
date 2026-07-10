import json
from datetime import datetime
from models import db


def get_next_session_number(user_id):
    query = """
        SELECT COUNT(*)
        FROM monitoring_sessions
        WHERE user_id = :user_id
    """

    result = db.session.execute(
        db.text(query),
        {"user_id": user_id}
    ).fetchone()

    return result[0] + 1


def save_monitoring_session(user_id, result):
    session_number = get_next_session_number(user_id)

    query = """
        INSERT INTO monitoring_sessions
        (user_id, session_number, total_skor, percentage, classification, abnormal_areas, created_at)
        VALUES
        (:user_id, :session_number, :total_skor, :percentage, :classification, :abnormal_areas, :created_at)
    """

    db.session.execute(
        db.text(query),
        {
            "user_id": user_id,
            "session_number": session_number,
            "total_skor": result["total_skor"],
            "percentage": result.get("percentage", 0),
            "classification": result["classification"],
            "abnormal_areas": json.dumps(result.get("abnormal_areas", [])),
            "created_at": datetime.now(),
        }
    )

    db.session.commit()


def get_monitoring_history(user_id):
    query = """
        SELECT
            session_number,
            total_skor,
            percentage,
            classification,
            abnormal_areas,
            created_at
        FROM monitoring_sessions
        WHERE user_id = :user_id
        ORDER BY session_number ASC
        LIMIT 7
    """

    rows = db.session.execute(
        db.text(query),
        {"user_id": user_id}
    ).fetchall()

    history = []

    for row in rows:
        history.append({
            "session_number": row[0],
            "total_skor": row[1],
            "percentage": row[2],
            "classification": row[3],
            "abnormal_areas": json.loads(row[4]) if row[4] else [],
            "created_at": row[5].strftime("%Y-%m-%d %H:%M:%S") if row[5] else None,
        })

    return history