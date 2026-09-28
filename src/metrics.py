"""Métriques géométriques par frame : EAR, MAR, pose 3D de tête.

Philosophie : des features INTERPRÉTABLES plutôt qu'un réseau boîte noire.
Chaque métrique a une définition publiée, des unités, et un sens physique —
c'est ce qui rend le système auditable et débogable.
"""
from __future__ import annotations

import cv2
import numpy as np

from .landmarks import LEFT_EYE, RIGHT_EYE, MOUTH, POSE_IDS


def eye_aspect_ratio(pts: np.ndarray, ids: list[int]) -> float:
    """EAR (Soukupová & Čech 2016) = (‖p2−p6‖+‖p3−p5‖) / (2‖p1−p4‖).

    Ratio de distances → invariant à l'échelle et à la distance caméra.
    ~0.30 yeux ouverts, ~0.10 fermés (varie selon les personnes, d'où la calibration).
    """
    p = pts[ids][:, :2]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return float((v1 + v2) / (2.0 * h + 1e-8))


def ear_both(pts: np.ndarray) -> float:
    """Moyenne des deux yeux : robuste si un œil est partiellement occulté."""
    return (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2.0


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    """MAR = ouverture verticale / largeur de bouche. > ~0.6 pendant ≥ 0.8 s ≈ bâillement."""
    v = np.linalg.norm(pts[MOUTH["top"], :2] - pts[MOUTH["bottom"], :2])
    h = np.linalg.norm(pts[MOUTH["left"], :2] - pts[MOUTH["right"], :2])
    return float(v / (h + 1e-8))


# Modèle 3D générique du visage (mm, repère tête) pour solvePnP
_FACE_3D = np.array([
    [0.0, 0.0, 0.0],        # bout du nez
    [0.0, -63.6, -12.5],    # menton
    [-43.3, 32.7, -26.0],   # coin externe œil gauche
    [43.3, 32.7, -26.0],    # coin externe œil droit
    [-28.9, -28.9, -24.1],  # coin gauche bouche
    [28.9, -28.9, -24.1],   # coin droit bouche
], dtype=np.float64)


def head_pose(pts: np.ndarray, frame_shape: tuple) -> tuple[float, float, float]:
    """Estime (yaw, pitch, roll) en degrés par solvePnP.

    Principe : on connaît la géométrie 3D approximative d'un visage humain
    (_FACE_3D) et on observe où ces 6 points tombent dans l'image (2D).
    solvePnP retrouve la rotation qui explique cette projection.
    Caméra approximée : focale = largeur d'image, centre optique = centre image
    (suffisant pour des angles, pas pour de la mesure métrique).
    """
    h, w = frame_shape[:2]
    image_pts = pts[POSE_IDS][:, :2].astype(np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_FACE_3D, image_pts, cam, np.zeros(4),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rot, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    pitch = float(np.degrees(np.arctan2(-rot[2, 0], sy)))
    yaw = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
    roll = float(np.degrees(np.arctan2(rot[2, 1], rot[2, 2])))
    # normalisation : 0° = face caméra
    if abs(roll) > 90:
        roll = roll - np.sign(roll) * 180
    return yaw, pitch, roll
