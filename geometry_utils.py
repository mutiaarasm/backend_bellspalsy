from __future__ import annotations

from math import atan2, degrees, hypot
from typing import Iterable

import numpy as np

EPSILON = 1e-8


def euclidean_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(hypot(float(point_b[0] - point_a[0]), float(point_b[1] - point_a[1])))


def line_slope(point_left: np.ndarray, point_right: np.ndarray) -> float:
    dx = float(point_right[0] - point_left[0])
    dy = float(point_right[1] - point_left[1])
    return dy / (dx + EPSILON)


def line_angle_degrees(point_left: np.ndarray, point_right: np.ndarray) -> float:
    return degrees(atan2(float(point_right[1] - point_left[1]), float(point_right[0] - point_left[0])))


def normalize_line_angle(angle: float) -> float:
    while angle > 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def mean_point(points: np.ndarray, indexes: Iterable[int]) -> np.ndarray:
    return np.mean(points[list(indexes)], axis=0)


def min_max_scale_landmarks(points: np.ndarray) -> np.ndarray:
    """Compatibility helper. Prefer isotropic_normalize_landmarks()."""
    result = points.copy().astype(np.float64)
    x_min, x_max = float(np.min(result[:, 0])), float(np.max(result[:, 0]))
    y_min, y_max = float(np.min(result[:, 1])), float(np.max(result[:, 1]))
    scale = max(x_max - x_min, y_max - y_min, EPSILON)
    center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
    return (result - center) / scale


def isotropic_normalize_landmarks(
    points: np.ndarray,
    left_eye_center: np.ndarray,
    right_eye_center: np.ndarray,
) -> np.ndarray:
    """Normalize by one physical scale so x/y geometry is not distorted."""
    center = (left_eye_center + right_eye_center) / 2.0
    interocular = euclidean_distance(left_eye_center, right_eye_center)
    if interocular <= EPSILON:
        raise ValueError("invalid_interocular_distance")
    return (points - center) / interocular


def remove_head_roll(
    points: np.ndarray,
    left_eye_center: np.ndarray,
    right_eye_center: np.ndarray,
) -> tuple[np.ndarray, float]:
    roll_angle = line_angle_degrees(left_eye_center, right_eye_center)
    center = (left_eye_center + right_eye_center) / 2.0
    radians = np.deg2rad(-roll_angle)
    rotation_matrix = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)],
    ])
    return ((points - center) @ rotation_matrix.T) + center, roll_angle


def asymmetry_index(left_value: float, right_value: float, *, activity_floor: float = 0.0) -> float:
    """Symmetric normalized difference with a floor for tiny/noisy motion.

    If both values are below activity_floor, the result is zero instead of
    exploding because of a tiny denominator. Movement quality is checked
    separately by the service.
    """
    left = abs(float(left_value))
    right = abs(float(right_value))
    if max(left, right) <= activity_floor:
        return 0.0
    denominator = max(left + right, 2.0 * activity_floor, EPSILON)
    return float(np.clip(abs(left - right) / denominator, 0.0, 1.0))


def weaker_side(left_value: float, right_value: float, tolerance: float = 0.02) -> str:
    left = float(left_value)
    right = float(right_value)
    scale = max(abs(left), abs(right), EPSILON)
    if abs(left - right) / scale <= tolerance:
        return "symmetric"
    return "left" if left < right else "right"
