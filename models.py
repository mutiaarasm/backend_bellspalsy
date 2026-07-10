from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    otp_code = db.Column(db.String(6), nullable=True)
    otp_expired_at = db.Column(db.DateTime, nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    avatar_url = db.Column(db.String(500), nullable=True)
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    reset_verified = db.Column(db.Boolean, default=False)

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