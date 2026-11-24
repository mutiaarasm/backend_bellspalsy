from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(
        db.Integer,
        primary_key=True,
    )

    name = db.Column(
        db.String(80),
        nullable=True,
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False,
        index=True,
    )

    # Akun dari Google tidak wajib memiliki password lokal.
    password_hash = db.Column(
        db.String(255),
        nullable=True,
    )

    # Nilai "sub" dari Google ID Token.
    # Ini adalah identitas permanen akun Google.
    google_sub = db.Column(
        db.String(255),
        unique=True,
        nullable=True,
        index=True,
    )

    # Nilai yang mungkin:
    # local, google, atau local_google.
    auth_provider = db.Column(
        db.String(20),
        nullable=False,
        default="local",
    )

    avatar_url = db.Column(
        db.String(500),
        nullable=True,
    )

    is_verified = db.Column(
        db.Boolean,
        nullable=False,
        default=False,
    )

    otp_code = db.Column(
        db.String(6),
        nullable=True,
    )

    otp_expired_at = db.Column(
        db.DateTime,
        nullable=True,
    )

    reset_verified = db.Column(
        db.Boolean,
        nullable=False,
        default=False,
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

class LoginHistory(db.Model):
    __tablename__ = "login_history"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    email = db.Column(db.String(120), nullable=False)

    ip_address = db.Column(db.String(50), nullable=True)

    device = db.Column(db.String(120), nullable=True)

    location = db.Column(db.String(120), nullable=True)

    status = db.Column(
        db.String(20),
        default="success"
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    title = db.Column(db.String(255))
    content = db.Column(db.Text)
    category = db.Column(db.String(100))

    image_url = db.Column(db.String(255))

    source_name = db.Column(db.String(255))
    source_url = db.Column(db.String(500))

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

class ExerciseVideo(db.Model):
    __tablename__ = 'exercise_videos'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    youtube_id = db.Column(db.String(50), nullable=False) # Contoh: chzl_w2AwxI
    description = db.Column(db.Text, nullable=True)
    order = db.Column(db.Integer, nullable=False) # Untuk urutan 1, 2, 3, 4, 5  

class MonitoringSession(db.Model):
    __tablename__ = "monitoring_sessions"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    session_number = db.Column(db.Integer, nullable=False)

    total_skor = db.Column(db.Integer, nullable=False)

    percentage = db.Column(db.Integer, nullable=False)

    classification = db.Column(
        db.String(100),
        nullable=False
    )

    abnormal_areas = db.Column(
        db.Text,
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )
class Hospital(db.Model):
    __tablename__ = "hospitals"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200))
    address = db.Column(db.Text)
    phone = db.Column(db.String(50))
    website = db.Column(db.String(255))
    operational_hours = db.Column(db.String(100))

    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)

class UserProgress(db.Model):
    __tablename__ = 'user_progress'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, nullable=False)
    session_number = db.Column(db.Integer, nullable=False)
    completed_at = db.Column(db.DateTime, nullable=False)
class DetectionHistory(db.Model):
    __tablename__ = "detection_histories"

    id = db.Column(
        db.Integer,
        primary_key=True,
        autoincrement=True,
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    mode = db.Column(
        db.String(20),
        nullable=False,
        index=True,
    )

    session_number = db.Column(
        db.Integer,
        nullable=True,
    )

    hb_grade = db.Column(
        db.Integer,
        nullable=True,
    )

    hb_roman = db.Column(
        db.String(5),
        nullable=True,
    )

    hb_label = db.Column(
        db.String(100),
        nullable=True,
    )

    classification_status = db.Column(
        db.String(30),
        nullable=False,
        default="classified",
    )

    face_function_score = db.Column(
        db.Float,
        nullable=True,
    )

    affected_side = db.Column(
        db.String(30),
        nullable=True,
    )

    regional_result = db.Column(
        db.JSON,
        nullable=False,
    )

    feature_values = db.Column(
        db.JSON,
        nullable=False,
    )

    algorithm_version = db.Column(
        db.String(50),
        nullable=False,
    )

    threshold_version = db.Column(
        db.String(50),
        nullable=False,
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )

    def to_dict(
        self,
        *,
        include_features: bool = True,
    ):
        data = {
            "id": self.id,
            "user_id": self.user_id,
            "mode": self.mode,
            "session_number": self.session_number,
            "hb_grade": self.hb_grade,
            "hb_roman": self.hb_roman,
            "hb_label": self.hb_label,
            "classification_status":
                self.classification_status,
            "face_function_score":
                self.face_function_score,

            # Sementara disediakan agar tampilan lama
            # yang masih membaca percentage tidak error.
            "percentage":
                self.face_function_score,

            "affected_side":
                self.affected_side,
            "regional_result":
                self.regional_result or {},
            "algorithm_version":
                self.algorithm_version,
            "threshold_version":
                self.threshold_version,
            "created_at": (
                self.created_at.isoformat()
                if self.created_at is not None
                else None
            ),
        }

        if include_features:
            data["feature_values"] = (
                self.feature_values or {}
            )

        return data 
     
class TherapyResult(db.Model):
    __tablename__ = "therapy_results"

    id = db.Column(
        db.Integer,
        primary_key=True,
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    day = db.Column(
        db.Integer,
        nullable=False,
        index=True,
    )

    exercise_key = db.Column(
        db.String(50),
        nullable=False,
        index=True,
    )

    video_id = db.Column(
        db.String(32),
        nullable=True,
    )

    classification = db.Column(
        db.String(30),
        nullable=False,
    )

    label = db.Column(
        db.String(255),
        nullable=True,
    )

    raw_metrics = db.Column(
        db.JSON,
        nullable=True,
    )

    algorithm_version = db.Column(
        db.String(60),
        nullable=True,
    )

    threshold_version = db.Column(
        db.String(60),
        nullable=True,
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True,
    )