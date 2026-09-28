"""Architecture du CNN de détection d'œil ouvert/fermé.

Chaque couche est commentée avec CE QU'ELLE APPREND et POURQUOI.

Un CNN traite l'image par couches successives : les premières apprennent des
motifs simples (bords, contrastes), les profondes les combinent en concepts
(paupière baissée, cils, iris visible). Deux couches denses transforment le tout
en décision.

═══════════════ CE QUI CHANGE PAR RAPPORT À LA v1 ═══════════════

1. ENTRÉE 64×64 AU LIEU DE 24×24.
   À 24 pixels de côté, un œil à moitié fermé et un œil ouvert diffèrent de
   quelques pixels sur la paupière. Or ce sont exactement les cas limites qui
   décident d'un micro-sommeil. Le coût CPU reste négligeable à cette taille de
   réseau, et c'est le changement qui rapporte le plus sur les cas difficiles.

2. L'ORDRE DES CLASSES N'EST PLUS SUPPOSÉ.
   La v1 déclarait `CLASS_NAMES = ["open", "closed"]` et lisait la colonne 1
   comme la probabilité "fermé". Or `ImageFolder` trie alphabétiquement, donc
   closed=0 et open=1 : la colonne 1 était la probabilité OUVERT. Les métriques
   publiées portaient sur la mauvaise classe. Ici on ne devine plus, l'index de
   "closed" est passé explicitement.

3. POOLING ADAPTATIF.
   La tête de classification a une taille fixe quelle que soit la résolution
   d'entrée : on peut expérimenter en 48 ou 96 sans retoucher le code.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Constantes partagées entre entraînement et inférence.
IMG_SIZE = 64
CROP_PAD = 0.35          # padding du crop autour des landmarks de l'œil
# Ordre imposé par ImageFolder (tri alphabétique). Ne pas réordonner :
# c'est cette liste qui a été fausse en v1.
CLASS_NAMES = ["closed", "open"]


def _block(cin: int, cout: int) -> nn.Sequential:
    """Conv → BatchNorm → ReLU → MaxPool. Le bloc de base de l'extracteur."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),      # stabilise et accélère l'entraînement
        nn.ReLU(inplace=True),     # non-linéarité
        nn.MaxPool2d(2),           # divise la résolution par deux
    )


class EyeCNN(nn.Module):
    """Image d'œil en niveaux de gris → 2 logits [closed, open]."""

    def __init__(self, img_size: int = IMG_SIZE, width: int = 32, dropout: float = 0.35):
        super().__init__()
        self.img_size = img_size
        self.width = width
        w = width

        self.features = nn.Sequential(
            _block(1, w),          # bords et contrastes            64 → 32
            _block(w, w * 2),      # formes : courbe de paupière     32 → 16
            _block(w * 2, w * 4),  # motifs : iris + blanc vs ligne  16 → 8
            # Pooling adaptatif : la tête voit toujours 3×3, quelle que soit
            # la résolution d'entrée choisie.
            nn.AdaptiveAvgPool2d(3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),               # anti-surapprentissage
            nn.Linear(w * 4 * 3 * 3, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(96, 2),                  # logits [closed, open]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @torch.no_grad()
    def predict_closed_proba(self, x: torch.Tensor, closed_idx: int = 0,
                             temperature: float = 1.0) -> torch.Tensor:
        """Probabilité de la classe 'fermé'.

        `closed_idx` est OBLIGATOIREMENT celui du `class_to_idx` du dataset —
        on ne le suppose pas. `temperature` applique la calibration apprise en
        validation : un réseau entraîné en cross-entropy est surconfiant, et
        c'est cette surconfiance qui produisait un signal en créneau,
        inexploitable pour mesurer une vitesse de fermeture.
        """
        logits = self.forward(x) / temperature
        return torch.softmax(logits, dim=1)[:, closed_idx]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
