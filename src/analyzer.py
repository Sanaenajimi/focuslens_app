"""Analyzer : orchestre landmarks → métriques → temporel, frame par frame.

Une seule classe à connaître pour utiliser FocusLens en bibliothèque :

    fl = FocusLens()
    for frame, t in frames:
        result = fl.process(frame, t)
    fl.finalize(t)

v2 — LE CNN PILOTE, L'EAR CONTRÔLE
Le réseau entraîné décide de l'état de l'œil. L'EAR géométrique est calculé à
chaque frame malgré tout, pour trois raisons :

  * la calibration personnelle reste géométrique ;
  * il sert de canal de contrôle (taux de désaccord entre les deux sources,
    reporté dans le bilan) ;
  * il prend le relais automatiquement si torch ou le checkpoint manquent,
    au lieu de faire planter la session.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from .config import FocusLensConfig
from .landmarks import FaceLandmarker, LEFT_EYE, RIGHT_EYE
from .metrics import ear_both, mouth_aspect_ratio, head_pose
from .temporal import TemporalAnalyzer


@dataclass
class FrameResult:
    t: float
    face_found: bool
    ear: float | None                    # TOUJOURS le vrai EAR géométrique
    mar: float | None
    yaw: float | None
    pitch: float | None
    roll: float | None
    score: float | None
    eye_state: str
    perclos: float
    blinks_per_min: float
    microsleep: bool
    phase: str
    thr_closed: float | None = None      # seuil appliqué au signal actif
    closed_duration: float = 0.0
    openness: float | None = None        # 1 − p(fermé) lissée, si le CNN pilote
    p_closed: float | None = None        # probabilité brute du réseau
    source: str = "ear"                  # source ayant décidé cette frame
    crop_quality: float | None = None    # fiabilité des patchs d'yeux (0-1)

    def to_row(self) -> dict:
        return asdict(self)

    def reason(self, cfg) -> str | None:
        """Justification textuelle de l'état courant, pour l'affichage live et
        le rapport. None si rien de notable à expliquer."""
        if not self.face_found:
            return "Visage non détecté, aucune mesure possible."

        # La preuve citée dépend de la source qui a réellement tranché.
        if self.source == "cnn" and self.p_closed is not None:
            proof = (f"le réseau donne {self.p_closed*100:.0f}% de probabilité "
                     f"'fermé' (seuil {(1-(self.thr_closed or 0))*100:.0f}%)")
        else:
            proof = (f"EAR {self.ear:.2f} sous le seuil {self.thr_closed:.2f}"
                     if self.ear is not None and self.thr_closed else "signal sous le seuil")

        if self.microsleep:
            return (f"SOMNOLENCE : yeux fermés depuis {self.closed_duration:.1f}s "
                    f"(seuil d'alerte {cfg.microsleep_s:.1f}s) — {proof}.")
        if self.eye_state == "closed" and self.closed_duration > 0:
            return (f"Yeux fermés depuis {self.closed_duration:.1f}s ({proof}), "
                    f"sous le seuil d'alerte de {cfg.microsleep_s:.1f}s : probable clignement.")
        if self.yaw is not None and abs(self.yaw) > cfg.yaw_away_deg:
            return f"Regard hors route : orientation {self.yaw:+.0f}° (seuil ±{cfg.yaw_away_deg:.0f}°)."
        if self.perclos > cfg.perclos_severity_threshold:
            return (f"Tendance de fond : {self.perclos*100:.0f}% du temps yeux fermés "
                    f"sur les {cfg.perclos_window_s:.0f} dernières secondes.")
        if self.crop_quality is not None and self.crop_quality < 0.25:
            return ("Patchs d'yeux peu exploitables (petits ou flous) : "
                    "la géométrie prend le relais sur ces images.")
        return None


class FocusLens:
    def __init__(self, cfg: FocusLensConfig | None = None):
        self.cfg = cfg or FocusLensConfig()
        self.landmarker = FaceLandmarker(self.cfg.min_detection_conf, self.cfg.min_tracking_conf)
        self.last_landmarks: np.ndarray | None = None

        # ── chargement du CNN, sans jamais faire tomber la session ──
        self.eye_cnn = None
        self.cnn_status = "Mode EAR géométrique demandé, aucun modèle chargé."
        if self.cfg.eye_source in ("cnn", "auto"):
            from .eye_cnn_infer import load_eye_cnn
            self.eye_cnn, self.cnn_status = load_eye_cnn()
            if self.eye_cnn is None:
                # Repli explicite : on le dit, on ne le cache pas.
                self.cfg.eye_source = "ear"

        thr = self.cfg.cnn_threshold
        if thr is None and self.eye_cnn is not None:
            thr = self.eye_cnn.threshold
        self.temporal = TemporalAnalyzer(self.cfg, cnn_threshold=thr)

    # ───────────────────────── prétraitement ─────────────────────────
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Redimensionnement borné : la précision des landmarks sature vers
        960 px, au-delà on paie juste en latence."""
        h, w = frame.shape[:2]
        if w > self.cfg.max_width:
            s = self.cfg.max_width / w
            frame = cv2.resize(frame, (self.cfg.max_width, int(h * s)),
                               interpolation=cv2.INTER_AREA)
        return frame

    # ───────────────────────── décision œil ─────────────────────────
    def _eye_signal(self, frame: np.ndarray, pts: np.ndarray, ear_geo: float):
        """Rend (openness, p_closed, quality, source).

        `openness` vaut None quand c'est l'EAR qui doit piloter la frame : la
        couche temporelle bascule alors seule sur ses seuils géométriques.
        """
        cfg = self.cfg
        if self.eye_cnn is None or cfg.eye_source == "ear":
            return None, None, None, "ear"

        crops = [self.eye_cnn.eye_crop(frame, pts[LEFT_EYE]),
                 self.eye_cnn.eye_crop(frame, pts[RIGHT_EYE])]
        p_closed, quality = self.eye_cnn.proba_closed(crops)

        if quality <= 0.05:
            # Aucun patch exploitable : la géométrie est le seul recours.
            return None, p_closed, quality, "ear"

        if cfg.eye_source == "auto" and cfg.cnn_uncertain_low <= p_closed <= cfg.cnn_uncertain_high:
            # Le réseau hésite. Plutôt que de trancher au hasard sur une frame
            # ambiguë, on laisse l'EAR décider celle-ci. Le réseau reprend la
            # main dès que sa probabilité ressort de la bande.
            return None, p_closed, quality, "ear"

        return 1.0 - p_closed, p_closed, quality, "cnn"

    # ───────────────────────── boucle ─────────────────────────
    def process(self, frame_bgr: np.ndarray, t: float) -> FrameResult:
        frame = self.preprocess(frame_bgr)
        pts = self.landmarker.process(frame)
        self.last_landmarks = pts

        if pts is None:
            state = self.temporal.update(t, False, None, None, None, None)
            return FrameResult(t, False, None, None, None, None, None,
                               state["score"], state["eye_state"], state["perclos"],
                               state["blinks_per_min"], state["microsleep"], state["phase"],
                               state.get("thr_closed"), state.get("closed_duration", 0.0),
                               None, None, "ear", None)

        # L'EAR géométrique est calculé À CHAQUE FRAME, quelle que soit la
        # source qui décide : il porte la calibration et le canal de contrôle.
        ear_geo = ear_both(pts)
        openness, p_closed, quality, source = self._eye_signal(frame, pts, ear_geo)

        mar = mouth_aspect_ratio(pts)
        yaw, pitch, roll = head_pose(pts, frame.shape)
        state = self.temporal.update(t, True, ear_geo, mar, yaw, pitch,
                                     openness=openness, source=source)

        return FrameResult(t, True, round(ear_geo, 4), round(mar, 4),
                           round(yaw, 1), round(pitch, 1), round(roll, 1),
                           state["score"], state["eye_state"], state["perclos"],
                           state["blinks_per_min"], state["microsleep"], state["phase"],
                           state.get("thr_closed"), state.get("closed_duration", 0.0),
                           state.get("openness"),
                           round(p_closed, 4) if p_closed is not None else None,
                           source,
                           round(quality, 3) if quality is not None else None)

    # ───────────────────────── bilan ─────────────────────────
    def source_summary(self) -> dict:
        """À afficher dans le rapport : qui a décidé, et les deux sources
        étaient-elles d'accord. C'est la traçabilité du système."""
        by = self.temporal.frames_by_source
        total = max(1, sum(by.values()))
        return {
            "mode": self.cfg.eye_source,
            "status": self.cnn_status,
            "frames_cnn": by.get("cnn", 0),
            "frames_ear": by.get("ear", 0),
            "share_cnn": round(by.get("cnn", 0) / total, 3),
            "agreement_rate": (round(self.temporal.agreement_rate, 3)
                               if self.temporal.agreement_rate is not None else None),
            "frames_compared": self.temporal.frames_compared,
        }

    def finalize(self, t_end: float) -> None:
        self.temporal.finalize(t_end)
        self.landmarker.close()
