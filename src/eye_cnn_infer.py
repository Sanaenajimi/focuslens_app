"""Pont entre le CNN entraîné et FocusLens.

Charge `train/models/eye_cnn.pt` et rend, pour une frame et ses landmarks, la
probabilité que l'œil soit fermé — accompagnée d'un indice de qualité du crop.

v2, trois corrections par rapport à la version précédente :

1. PLUS DE PSEUDO-EAR. On rendait autrefois `0.34 - p*0.24` pour faire passer la
   sortie du réseau pour un EAR. C'était doublement faux : ça inventait une
   grandeur géométrique qui n'en était pas une, et ça écrasait la dynamique du
   signal. On rend maintenant la probabilité brute ; c'est la couche temporelle
   qui la lisse et lui applique un seuil calibré.

2. REPLI AU LIEU DU CRASH. Si torch n'est pas installé ou si le checkpoint est
   absent, on n'explose plus au chargement : `load_eye_cnn()` rend None avec un
   message, et FocusLens bascule sur l'EAR géométrique en le signalant.

3. CADRAGE COHÉRENT. Le padding du crop est lu depuis le checkpoint, donc
   identique à celui utilisé pour générer les images d'entraînement. C'était la
   principale source d'écart entre les métriques de test et le comportement
   réel en voiture.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    _TORCH_OK = True
    _TORCH_ERR = ""
except ImportError as exc:  # pragma: no cover - dépend de l'environnement
    _TORCH_OK = False
    _TORCH_ERR = str(exc)

_MODEL_PATH = Path(__file__).resolve().parents[1] / "train" / "models" / "eye_cnn.pt"

# Cadrage par défaut si le checkpoint ne le précise pas (modèles v1).
DEFAULT_CROP_PAD = 0.35


class EyeStateCNN:
    """Charge le modèle une fois, prédit la probabilité 'fermé' d'un patch d'œil."""

    def __init__(self, model_path: Path | None = None, threads: int = 2):
        if not _TORCH_OK:
            raise ImportError(f"PyTorch requis pour le mode CNN ({_TORCH_ERR}). "
                              "pip install torch torchvision")
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
        from models.eye_cnn import EyeCNN

        path = model_path or _MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {path}\n"
                "Entraîne-le d'abord : python train/train_eye_cnn.py --data train/data")

        # Sur CPU, laisser torch prendre tous les cœurs fait chuter le fps de la
        # boucle vidéo. Deux threads suffisent pour un batch de deux patchs.
        torch.set_num_threads(max(1, threads))

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.img_size = int(ckpt.get("img_size", 64))
        self.width = int(ckpt.get("width", 32))
        self.model = EyeCNN(img_size=self.img_size, width=self.width)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        # ImageFolder trie les classes alphabétiquement (closed=0, open=1). On ne
        # DEVINE pas l'index : on le lit dans le checkpoint. C'est exactement le
        # bug qui inversait les métriques dans la version précédente.
        c2i = ckpt.get("class_to_idx", {"closed": 0, "open": 1})
        self.closed_idx = int(c2i.get("closed", 0))

        # Seuil calibré à l'entraînement (point de la ROC atteignant le rappel
        # cible sur la classe "fermé"), et température de calibration.
        self.threshold: float | None = ckpt.get("threshold")
        self.temperature: float = float(ckpt.get("temperature", 1.0))
        self.crop_pad: float = float(ckpt.get("crop_pad", DEFAULT_CROP_PAD))
        self.metrics: dict = ckpt.get("metrics", {})

    # ───────────────────────── découpe ─────────────────────────
    def eye_crop(self, frame_bgr: np.ndarray, eye_pts: np.ndarray,
                 pad: float | None = None) -> np.ndarray | None:
        """Découpe la région de l'œil autour de ses landmarks, en carré.
        Le padding par défaut vient du checkpoint pour rester cohérent avec
        les images sur lesquelles le réseau a appris."""
        pad = self.crop_pad if pad is None else pad
        xs, ys = eye_pts[:, 0], eye_pts[:, 1]
        w = xs.max() - xs.min()
        h = ys.max() - ys.min()
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        half = max(w, h) * (1 + pad) / 2
        x1, y1 = int(cx - half), int(cy - half)
        x2, y2 = int(cx + half), int(cy + half)
        H, W = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 6 or y2 - y1 < 6:
            return None
        return frame_bgr[y1:y2, x1:x2]

    def _to_tensor(self, crops: list[np.ndarray]):
        """Prétraitement identique à celui de l'entraînement : gris, resize,
        normalisation [-1, 1]. Fait en OpenCV plutôt qu'en torchvision pour
        éviter un aller-retour PIL par frame."""
        batch = np.empty((len(crops), 1, self.img_size, self.img_size), dtype=np.float32)
        for i, c in enumerate(crops):
            g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            batch[i, 0] = (g.astype(np.float32) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(batch)

    @staticmethod
    def crop_quality(crop: np.ndarray | None) -> float:
        """Indice 0-1 de fiabilité du patch : taille suffisante et image nette.
        Un patch minuscule ou flou (mouvement, mise au point) donne une
        prédiction dont il ne faut pas se contenter."""
        if crop is None or crop.size == 0:
            return 0.0
        h, w = crop.shape[:2]
        size_ok = min(1.0, min(h, w) / 24.0)
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Variance du laplacien : mesure classique de netteté.
        sharp = cv2.Laplacian(g, cv2.CV_64F).var()
        sharp_ok = min(1.0, sharp / 60.0)
        return float(size_ok * 0.5 + sharp_ok * 0.5)

    # ───────────────────────── inférence ─────────────────────────
    def proba_closed(self, crops: list[np.ndarray | None]) -> tuple[float, float]:
        """Rend (probabilité 'fermé', qualité moyenne des patchs).

        Les deux yeux sont pondérés par la qualité de leur patch plutôt que
        moyennés à parts égales : si un œil est occulté par la monture ou coupé
        par le bord de l'image, son avis compte moins.
        """
        pairs = [(c, self.crop_quality(c)) for c in crops]
        valid = [(c, q) for c, q in pairs if c is not None and q > 0.05]
        if not valid:
            return 0.0, 0.0

        with torch.inference_mode():
            batch = self._to_tensor([c for c, _ in valid])
            logits = self.model(batch) / self.temperature
            p = torch.softmax(logits, dim=1)[:, self.closed_idx].numpy()

        qs = np.array([q for _, q in valid], dtype=np.float32)
        p_closed = float(np.average(p, weights=qs))
        return p_closed, float(qs.mean())


def load_eye_cnn(model_path: Path | None = None) -> tuple[EyeStateCNN | None, str]:
    """Chargement tolérant : rend (modèle, message). Le modèle vaut None si le
    mode CNN n'est pas disponible — l'appelant bascule alors sur l'EAR."""
    try:
        cnn = EyeStateCNN(model_path)
    except Exception as exc:
        return None, f"Mode CNN indisponible ({exc}). Repli sur l'EAR géométrique."
    thr = cnn.threshold
    thr_txt = f"seuil calibré {thr:.3f}" if thr is not None else "seuil non calibré"
    return cnn, (f"CNN chargé : {cnn.img_size}×{cnn.img_size}, {thr_txt}, "
                 f"température {cnn.temperature:.2f}.")
