from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for, session
from models import ExerciseVideo, Hospital, db, User, LoginHistory, Article, MonitoringSession
from user_agents import parse
from models import db
from geopy.distance import geodesic
from functools import wraps
from dotenv import load_dotenv


from detection_service import analyze_detection_frames, save_user_progress
from rule_based import classify_result
from monitoring_service import (
    get_next_session_number,
    save_monitoring_session,
    get_monitoring_history
)
import numpy as np
import random
import requests
from datetime import datetime, timedelta
import os
import cv2
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity
)
load_dotenv()

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_ARTIKEL_DIR = os.path.join(BASE_DIR, "static", "uploads")


os.makedirs(STATIC_ARTIKEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=7)

app.config["MAIL_SERVER"] = os.getenv("MAIL_SERVER")
app.config["MAIL_PORT"] = int(os.getenv("MAIL_PORT", 587))
app.config["MAIL_USE_TLS"] = os.getenv("MAIL_USE_TLS", "True") == "True"
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.getenv("MAIL_DEFAULT_SENDER")



mail = Mail(app)

serializer = URLSafeTimedSerializer(app.config["JWT_SECRET_KEY"])

jwt = JWTManager(app)
@jwt.invalid_token_loader
def invalid_token_callback(err):
    return jsonify({"message": "Invalid token", "detail": err}), 422

@jwt.unauthorized_loader
def missing_token_callback(err):
    return jsonify({"message": "Missing Authorization Header", "detail": err}), 401

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({"message": "Token expired"}), 401

# pakai SQL (SQLite dulu)
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root@localhost:3306/bellspalsy_db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
migrate = Migrate(app, db)

@app.get("/")
def home():
    return "Flask backend + DB jalan 🚀"

@app.post("/auth/register")
def register():
    data = request.get_json(silent=True) or {}

    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"message": "email dan password wajib diisi"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"message": "email sudah terdaftar"}), 409

    otp_code = f"{random.randint(0, 9999):04d}"

    user = User(
        name=name,
        email=email,
        password_hash=generate_password_hash(password),
        is_verified=False,
        otp_code=otp_code,
        otp_expired_at=datetime.utcnow() + timedelta(minutes=5)
    )

    db.session.add(user)
    db.session.commit()

    msg = Message(
        subject="Kode Verifikasi Akun",
        recipients=[email]
    )

    msg.body = f"""
Halo {name or 'User'},

Kode OTP verifikasi akun kamu adalah:

{otp_code}

Kode ini berlaku 5 menit.
Jangan berikan kode ini ke siapa pun.
"""

    mail.send(msg)

    return jsonify({
        "message": "register berhasil, cek email untuk kode OTP"
    }), 201

@app.post("/auth/verify")
def verify_email_otp():
    data = request.get_json(silent=True) or {}

    email = (data.get("email") or "").strip().lower()
    otp = (data.get("otp") or "").strip()

    if not email or not otp:
        return jsonify({"message": "email dan otp wajib diisi"}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"message": "user tidak ditemukan"}), 404

    if user.is_verified:
        return jsonify({"message": "email sudah diverifikasi"}), 200

    if not user.otp_code or not user.otp_expired_at:
        return jsonify({"message": "otp tidak tersedia, silakan daftar ulang atau kirim ulang otp"}), 400

    if datetime.utcnow() > user.otp_expired_at:
        return jsonify({"message": "otp sudah expired"}), 400

    if user.otp_code != otp:
        return jsonify({"message": "otp salah"}), 400

    user.is_verified = True
    user.otp_code = None
    user.otp_expired_at = None
    db.session.commit()

    return jsonify({"message": "email berhasil diverifikasi"}), 200


@app.post("/auth/login")
def login():

    data = request.get_json(silent=True) or {}

    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"message": "email atau password salah"}), 401

    if not user.is_verified:
        return jsonify({"message": "email belum diverifikasi"}), 403

    access_token = create_access_token(identity=str(user.id))

    user_agent_string = request.headers.get("User-Agent")

    ua = parse(user_agent_string)

    device = f"{ua.browser.family} - {ua.os.family}"

    if "Other" in device:
        device = "Mobile Device"
    ip_address = request.remote_addr

    history = LoginHistory(
        user_id=user.id,
        email=user.email,
        ip_address=ip_address,
        device=device,
        location=ip_address,
        status="success",
    )

    db.session.add(history)
    db.session.commit()

    # Kembalikan user ID, email, dan token
    return jsonify({
        "access_token": access_token,
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email
        }
    }), 200

@app.get("/auth/login-history")
@jwt_required()
def login_history():

    user_id = int(get_jwt_identity())

    histories = (
        LoginHistory.query
        .filter_by(user_id=user_id)
        .order_by(LoginHistory.created_at.desc())
        .all()
    )

    return jsonify([
    {
        "device": h.device,
        "location": h.location,
        "ip": h.ip_address,
        "status": h.status,
        "time": h.created_at.isoformat() + "Z"
    }
    for h in histories
])


@app.get("/me")
@jwt_required()
def me():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    if not user:
        return jsonify({"message": "user tidak ditemukan"}), 404

    return jsonify({
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "avatar_url": user.avatar_url,
        "created_at": user.created_at.isoformat()
    }), 200

@app.put("/profile")
@jwt_required()
def update_profile():

    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    data = request.get_json(silent=True) or {}

    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    avatar_url = (data.get("avatar_url") or "").strip()

    email_changed = False

    if name:
        user.name = name

    if email and email != user.email:

        if User.query.filter_by(email=email).first():
            return jsonify({"message": "email sudah digunakan"}), 409

        user.email = email
        user.is_verified = False

        # buat OTP baru
        otp_code = f"{random.randint(0, 9999):04d}"

        user.otp_code = otp_code
        user.otp_expired_at = datetime.utcnow() + timedelta(minutes=5)

        email_changed = True

    user.avatar_url = avatar_url if avatar_url else None

    db.session.commit()

    # kirim email OTP jika email berubah
    if email_changed:

        msg = Message(
            subject="Kode OTP Email Baru",
            recipients=[user.email]
        )

        msg.body = f"""
Halo {user.name}

Kode OTP verifikasi email baru kamu:

{otp_code}

Kode berlaku 5 menit.
"""

        mail.send(msg)

    return jsonify({
        "message": "profil berhasil diupdate",
        "email_changed": email_changed
    }), 200

@app.post("/profile/avatar")
@jwt_required()
def upload_avatar():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    if not user:
        return jsonify({"message": "user tidak ditemukan"}), 404

    if "avatar" not in request.files:
        return jsonify({"message": "file avatar tidak ditemukan"}), 400

    file = request.files["avatar"]
    if file.filename == "":
        return jsonify({"message": "nama file kosong"}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        return jsonify({"message": "format file harus jpg/png/webp"}), 400

    new_name = f"user_{user_id}{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], new_name)
    file.save(save_path)

    # URL yang bisa diakses Flutter
    avatar_url = f"/uploads/{new_name}"
    user.avatar_url = avatar_url
    db.session.commit()

    return jsonify({"message": "avatar berhasil diupload", "avatar_url": avatar_url}
    ), 200

@app.get("/uploads/<path:filename>")
def get_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/validate-face", methods=["POST"])
def validate_face():
    if "image" not in request.files:
        return jsonify({"error": "Image file is required"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400
    
@app.post("/auth/forgot-password")
def forgot_password():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"success": False, "message": "Email tidak terdaftar"}), 404

    otp_code = f"{random.randint(0, 9999):04d}"
    user.otp_code = otp_code
    user.otp_expired_at = datetime.utcnow() + timedelta(minutes=5)
    user.reset_verified = False
    db.session.commit()

    # Kirim email OTP
    msg = Message(
        subject="Reset Password OTP",
        recipients=[email],
        body=f"Kode OTP kamu: {otp_code} (berlaku 5 menit)"
    )
    mail.send(msg)

    return jsonify({"success": True, "message": "OTP telah dikirim ke email"})

@app.post("/auth/verify-reset-otp")
def verify_reset_otp():
    data = request.get_json() or {}

    email = (data.get("email") or "").strip().lower()
    otp = (data.get("otp") or "").strip()

    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({
            "success": False,
            "message": "User tidak ditemukan"
        }), 404

    if not user.otp_code or not user.otp_expired_at:
        return jsonify({
            "success": False,
            "message": "OTP tidak tersedia"
        }), 400

    if datetime.utcnow() > user.otp_expired_at:
        return jsonify({
            "success": False,
            "message": "OTP expired"
        }), 400

    if user.otp_code != otp:
        return jsonify({
            "success": False,
            "message": "OTP salah"
        }), 400

    user.reset_verified = True
    db.session.commit()

    return jsonify({
        "success": True,
        "message": "OTP valid"
    })

@app.post("/auth/reset-password")
def reset_password():
    data = request.get_json() or {}

    email = (data.get("email") or "").strip().lower()
    new_password = data.get("password") or ""

    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({
            "success": False,
            "message": "User tidak ditemukan"
        }), 404

    if not user.reset_verified:
        return jsonify({
            "success": False,
            "message": "OTP belum diverifikasi"
        }), 403

    if len(new_password) < 8:
        return jsonify({
            "success": False,
            "message": "Password minimal 8 karakter"
        }), 400

    user.password_hash = generate_password_hash(new_password)

    # Hapus OTP setelah password berhasil diubah
    user.otp_code = None
    user.otp_expired_at = None
    user.reset_verified = False

    db.session.commit()

    return jsonify({
        "success": True,
        "message": "Password berhasil diubah"
    }), 200
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH")


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login", next=request.path))
        return f(*args, **kwargs)
    return wrapper


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if session.get("admin_logged_in"):
        return redirect(url_for("admin_articles"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
            session.clear()
            session["admin_logged_in"] = True
            session["admin_username"] = username

            next_url = request.args.get("next") or url_for("admin_articles")
            return redirect(next_url)

        error = "Username atau password salah"

    return render_template("admin_login.html", error=error)


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))

# ==========================================
# ROUTE WEB ADMIN (UNTUK BROWSER)
# ==========================================
@app.route('/admin/articles', methods=['GET', 'POST'])
@admin_required
def admin_articles():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        category = request.form.get('category')
        file = request.files.get('image')
        source_name = request.form.get('source_name')
        source_url = request.form.get('source_url')

        filename = None
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            # Simpan ke folder static/uploads
            file.save(os.path.join(STATIC_ARTIKEL_DIR, filename))
        
        # Simpan data ke MySQL
        new_article = Article(
        title=title,
        content=content,
        category=category,
        image_url=filename,
        source_name=source_name,
        source_url=source_url
    )
        db.session.add(new_article)
        db.session.commit()
        return redirect(url_for('admin_articles'))

    # Tampilkan list artikel di bawah form
    articles = Article.query.order_by(Article.created_at.desc()).all()
    return render_template('admin_artikel.html', articles=articles)

@app.route('/admin/articles/delete/<int:id>')
@admin_required
def delete_article(id):

    article = Article.query.get_or_404(id)

    db.session.delete(article)
    db.session.commit()

    return redirect(url_for('admin_articles'))

# ==========================================
# ROUTE API ARTIKEL (UNTUK FLUTTER)
# ==========================================
@app.route('/api/articles', methods=['GET'])
def get_articles_api():
    articles = Article.query.all()

    results = []
    for art in articles:
       results.append({
            "id": art.id,
            "title": art.title,
            "content": art.content,
            "category": art.category,
            "image_url": f"{request.host_url}static/uploads/{art.image_url}" if art.image_url else None,
            "source_name": art.source_name,
            "source_url": art.source_url,
            "created_at": art.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    return jsonify(results)

@app.route('/admin/articles/edit/<int:id>', methods=['GET', 'POST'])
def edit_article(id):

    article = Article.query.get_or_404(id)

    if request.method == 'POST':

        article.title = request.form.get('title')
        article.category = request.form.get('category')
        article.content = request.form.get('content')

        article.source_name = request.form.get('source_name')
        article.source_url = request.form.get('source_url')

        file = request.files.get('image')

        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file.save(os.path.join(STATIC_ARTIKEL_DIR, filename))
            article.image_url = filename

        db.session.commit()

        return redirect(url_for('admin_articles'))

    return render_template(
        'edit_article.html',
        article=article
    )

# Endpoint untuk melayani file gambar artikel di browser/app
@app.route('/static/uploads/<path:filename>')
def get_static_upload(filename):
    return send_from_directory(STATIC_ARTIKEL_DIR, filename)

@app.route('/admin/videos')
@admin_required
def admin_videos():
    all_videos = ExerciseVideo.query.order_by(ExerciseVideo.order).all()
    return render_template('admin_videos.html', videos=all_videos)

# Route untuk proses tambah video
@app.route('/admin/videos/add', methods=['POST'])
@admin_required
def add_video():
    title = request.form.get('title')
    yt_id = request.form.get('youtube_id')
    desc = request.form.get('description')
    order = request.form.get('order')

    new_video = ExerciseVideo(title=title, youtube_id=yt_id, description=desc, order=order)
    db.session.add(new_video)
    db.session.commit()
    return redirect(url_for('admin_videos'))

import traceback

@app.post("/api/detection/analyze")
@jwt_required()
def analyze_detection():
    try:
        data = request.get_json(silent=True) or {}
        frames = data.get("frames")

        if not frames:
            return jsonify({
                "success": False,
                "message": "Frames kosong"
            }), 400

        mode = data.get("mode", "quick")
        user_id = int(get_jwt_identity())

        if len(frames) > 20:
            return jsonify({
            "success": False,
            "message": "Too many frames (limit 20)"
        }), 400

        movement_features = analyze_detection_frames(frames)
        result = classify_result(movement_features)

        if mode == "monitoring":
            save_monitoring_session(user_id, result)

            session_number = get_next_session_number(user_id) - 1
            save_user_progress(user_id, session_number)

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/monitoring/history", methods=["GET"])
@jwt_required()
def monitoring_history():
    try:
        user_id = int(get_jwt_identity())
        history = get_monitoring_history(user_id)
        
        # Pastikan history berbentuk list biasa, bukan objek query SQL raw
        return jsonify({
            "success": True, 
            "data": history
        }), 200
    except Exception as e:
        # Menampilkan error asli di terminal Flask agar kelihatan salahnya di mana
        print(f"Error di monitoring_history: {str(e)}") 
        return jsonify({
            "success": False, 
            "message": f"Terjadi kesalahan di server: {str(e)}"
        }), 500
    
@app.route("/api/monitoring/reset", methods=["DELETE"])
@jwt_required()
def reset_monitoring():
    user_id = int(get_jwt_identity())
    MonitoringSession.query.filter_by(user_id=user_id).delete()
    db.session.commit()
    return jsonify({"success": True, "message": "Monitoring program reset successfully"})

@app.route("/api/monitoring/repeat", methods=["POST"])
@jwt_required()
def repeat_monitoring():
    user_id = int(get_jwt_identity())

    # Hapus semua session lama user
    MonitoringSession.query.filter_by(user_id=user_id).delete()
    db.session.commit()

    return jsonify({"success": True, "message": "Monitoring session reset, ready to repeat"})
    
@app.route('/api/nearby-healthcare', methods=['GET'])
def nearby_healthcare():

    lat = request.args.get('lat', type=float)
    lng = request.args.get('lng', type=float)

    if not lat or not lng:
        return jsonify({
            "success": False,
            "message": "Latitude and longitude required"
        }), 400

    hospitals = Hospital.query.all()

    results = []

    for hospital in hospitals:

        distance = geodesic(
            (lat, lng),
            (hospital.latitude, hospital.longitude)
        ).km

        results.append({
            "id": hospital.id,
            "name": hospital.name,
            "address": hospital.address,
            "phone": hospital.phone,
            "website": hospital.website,
            "operational_hours": hospital.operational_hours,
            "lat": hospital.latitude,
            "lng": hospital.longitude,
            "distance": round(distance, 2)
        })

    results.sort(key=lambda x: x["distance"])

    nearby_results = [
    item for item in results
    if item["distance"] <= 5
]

    return jsonify({
        "success": True,
        "data": results[:3]  # tampilkan 3 terdekat
    })

if __name__ == "__main__":
    print("MENJALANKAN FLASK SERVER...")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )

