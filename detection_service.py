import base64
from datetime import datetime
import math
import cv2
import numpy as np
import mediapipe as mp
from models import db, UserProgress

from mediapipe.tasks import python
from mediapipe.tasks.python import vision



MODEL_PATH = "models/face_landmarker.task"

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

LEFT_EYEBROW = 105
RIGHT_EYEBROW = 334

LEFT_MOUTH = 61
RIGHT_MOUTH = 291


base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
)

landmarker = vision.FaceLandmarker.create_from_options(options)



def analyze_detection_frames(frames):
    result = {}

    for movement, frame_list in frames.items():
        features_list = []

        for frame_base64 in frame_list:
            image = decode_base64_image(frame_base64)

            if image is None:
                continue

            features = analyze_single_frame(image)

            if features is not None:
                features_list.append(features)

        if len(features_list) == 0:
            result[movement] = {
                "mata": 0,
                "alis": 0,
                "mulut": 0,
            }
        else:
            result[movement] = aggregate_features(features_list)

    return result


def decode_base64_image(base64_string):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        image_bytes = base64.b64decode(base64_string)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        return image
    except Exception as e:
        print("Decode image error:", e)
        return None


def analyze_single_frame(image_bgr):
    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb,
        )

        detection_result = landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return None

        landmarks = detection_result.face_landmarks[0]

        normalized = normalize_landmarks(landmarks)

        f_mata = calculate_eye_asymmetry(normalized)
        f_alis = calculate_angle_asymmetry(
            normalized[RIGHT_EYEBROW],
            normalized[LEFT_EYEBROW],
        )
        f_mulut = calculate_angle_asymmetry(
            normalized[RIGHT_MOUTH],
            normalized[LEFT_MOUTH],
        )

        return {
            "mata": f_mata,
            "alis": f_alis,
            "mulut": f_mulut,
        }

    except Exception as e:
        print("Analyze frame error:", e)
        return None


def normalize_landmarks(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    normalized = []

    for lm in landmarks:
        x_new = (lm.x - x_min) / (x_max - x_min)
        y_new = (lm.y - y_min) / (y_max - y_min)
        normalized.append((x_new, y_new))

    return normalized


def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_eye_asymmetry(landmarks):
    d_left = euclidean_distance(
        landmarks[LEFT_EYE_TOP],
        landmarks[LEFT_EYE_BOTTOM],
    )

    d_right = euclidean_distance(
        landmarks[RIGHT_EYE_TOP],
        landmarks[RIGHT_EYE_BOTTOM],
    )

    return abs(d_left - d_right)


def calculate_angle_asymmetry(point_right, point_left):
    x1, y1 = point_right
    x2, y2 = point_left

    delta_y = abs(y2 - y1)
    delta_x = abs(x2 - x1)

    if delta_x == 0:
        return 0

    angle = math.degrees(math.atan(delta_y / delta_x))

    return angle

def aggregate_features(features_list):
    return {
        "mata": max(item["mata"] for item in features_list),
        "alis": max(item["alis"] for item in features_list),
        "mulut": max(item["mulut"] for item in features_list),
    }
def save_user_progress(user_id, session_number):
    try:
        progress = UserProgress(
            user_id=user_id,
            session_number=session_number,
            completed_at=datetime.utcnow()
        )

        db.session.add(progress)
        db.session.commit()

    except Exception as e:
        db.session.rollback()
        print("SAVE PROGRESS ERROR:", e)