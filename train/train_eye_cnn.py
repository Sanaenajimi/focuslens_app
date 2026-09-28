"""
Entraînement d'un CNN de détection d'œil ouvert / fermé.

Commande typique :
    python train/train_eye_cnn.py --data train/data --epochs 25

======================================================================
OBJECTIF DU SCRIPT
======================================================================

Ce script entraîne un réseau de neurones convolutif (CNN) capable de
classifier une image d'œil en deux classes :

    0 → open   = œil ouvert
    1 → closed = œil fermé

Le pipeline complet est :

    Images
       ↓
    Chargement du dataset
       ↓
    Identification des sujets/personnes
       ↓
    Séparation TRAIN / VALIDATION / TEST par sujet
       ↓
    Data augmentation uniquement sur TRAIN
       ↓
    CNN
       ↓
    Logits
       ↓
    Cross Entropy Loss
       ↓
    Backpropagation
       ↓
    Mise à jour des poids avec AdamW
       ↓
    Répétition pendant plusieurs epochs
       ↓
    Meilleur modèle selon la validation
       ↓
    Calibration de température
       ↓
    Choix d'un seuil adapté au problème
       ↓
    Évaluation finale sur des sujets jamais vus
       ↓
    Sauvegarde du modèle + métriques + graphiques


======================================================================
1. POURQUOI UNE DÉCOUPE PAR SUJET ?
======================================================================

Les datasets d'yeux comme MRL Eye Dataset sont souvent constitués de
séquences contenant beaucoup de frames très similaires provenant
du même œil et de la même personne.

Exemple :

    Personne A
        ├── frame_001
        ├── frame_002
        ├── frame_003
        ├── frame_004
        └── frame_005

Une mauvaise stratégie serait :

    TRAIN :
        frame_001
        frame_003
        frame_005

    TEST :
        frame_002
        frame_004

Le modèle aurait pratiquement déjà vu le même œil pendant
l'entraînement.

On obtiendrait alors potentiellement un score très élevé, mais ce score
mesurerait surtout la capacité du modèle à mémoriser les caractéristiques
de personnes déjà vues.

C'est une forme de DATA LEAKAGE / FUITE DE DONNÉES.

La bonne stratégie est donc :

    TRAIN :
        Personnes A, B, C, D...

    VALIDATION :
        Personnes E, F...

    TEST :
        Personnes G, H...

Ainsi, une personne présente dans TEST n'a jamais été utilisée pour
apprendre les poids du CNN.

Le score de test devient donc beaucoup plus représentatif de la capacité
du modèle à généraliser à un nouveau conducteur.


======================================================================
2. DATA AUGMENTATION
======================================================================

Les images d'entraînement sont volontairement modifiées :

    - rotation
    - déplacement
    - flip horizontal
    - changement de luminosité
    - changement de contraste
    - flou
    - crop aléatoire
    - effacement partiel

Pourquoi ?

Parce que dans une vraie caméra :

    - la tête peut être inclinée
    - l'œil peut être légèrement décentré
    - l'éclairage peut changer
    - la caméra peut bouger
    - l'image peut être floue
    - des lunettes ou cheveux peuvent cacher une partie de l'œil

Le CNN doit donc apprendre les caractéristiques importantes de
l'œil et non mémoriser une apparence exacte.


======================================================================
3. CALIBRATION DE TEMPÉRATURE
======================================================================

Un CNN entraîné avec CrossEntropy peut devenir très confiant.

Exemple :

    P(open)   = 0.001
    P(closed) = 0.999

Même lorsqu'il fait parfois des erreurs.

La calibration de température consiste à apprendre un scalaire T :

    logits_calibrés = logits / T

On ne modifie PAS les poids du CNN.

On modifie uniquement la manière dont on transforme les logits
en probabilités.

Cela permet d'obtenir une sortie plus exploitable comme signal continu.


======================================================================
4. CHOIX DU SEUIL
======================================================================

Un seuil de 0.5 est souvent utilisé par défaut :

    P(closed) >= 0.5 → closed
    P(closed) <  0.5 → open

Mais 0.5 n'est pas forcément adapté à notre application.

Ici, rater un œil réellement fermé peut être plus problématique
qu'une fausse alarme.

On cherche donc sur la VALIDATION un seuil permettant d'atteindre
un rappel cible sur la classe "closed".

Par défaut :

    target_recall = 0.98

Cela signifie :

    détecter au moins 98 % des yeux réellement fermés
    sur la validation.

Une fois le seuil choisi, il est sauvegardé dans le checkpoint.


======================================================================
5. MÉTRIQUES
======================================================================

La classe POSITIVE est explicitement :

    closed = œil fermé

On calcule notamment :

    Accuracy
    AUROC
    Precision closed
    Recall closed
    F1 closed
    Miss rate closed
    False alarm rate
    Confusion matrix

Le "miss rate closed" représente la proportion d'yeux fermés
que le modèle a ratés.

======================================================================
"""


# =====================================================================
# IMPORTS
# =====================================================================

# Permet notamment une meilleure gestion des annotations de types.
from __future__ import annotations

# argparse permet de récupérer les paramètres passés depuis le terminal.
#
# Exemple :
#
# python train/train_eye_cnn.py --data train/data --epochs 25
#
# Le programme pourra récupérer :
#
# args.data   = "train/data"
# args.epochs = 25
import argparse

# Permet de sauvegarder les métriques dans un fichier JSON.
import json

# Permet d'utiliser les expressions régulières.
# Ici, elles servent notamment à extraire l'identifiant du sujet
# depuis le nom du fichier.
import re

# Permet de modifier le chemin de recherche des modules Python.
import sys

# defaultdict est un dictionnaire pratique pour regrouper les images
# par sujet.
from collections import defaultdict

# Path permet de manipuler les chemins de fichiers proprement.
from pathlib import Path

# NumPy :
# - tableaux numériques
# - génération aléatoire
# - calculs statistiques
# - manipulation des indices
import numpy as np

# PyTorch :
# framework utilisé pour construire et entraîner le CNN.
import torch

# torch.nn contient les fonctions/modules de deep learning.
import torch.nn as nn

# DataLoader :
# permet de fournir les images au réseau par lots (batches).
#
# Subset :
# permet de sélectionner seulement certaines images d'un dataset.
from torch.utils.data import DataLoader, Subset

# Transformations d'images de torchvision.
from torchvision import transforms

# ImageFolder permet de construire automatiquement un dataset
# à partir d'une structure de dossiers.
#
# Exemple :
#
# data/
# ├── closed/
# └── open/
#
# ImageFolder associera automatiquement chaque dossier à une classe.
from torchvision.datasets import ImageFolder


# =====================================================================
# IMPORT DU MODÈLE CNN
# =====================================================================

# On ajoute le dossier contenant ce script dans les chemins Python.
#
# __file__
#     → chemin du fichier actuel
#
# Path(__file__).resolve()
#     → chemin absolu
#
# .parent
#     → dossier contenant le fichier
#
# str(...)
#     → conversion en chaîne
#
# sys.path.insert(0, ...)
#     → permet à Python de chercher les modules dans ce dossier.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

# EyeCNN est l'architecture du réseau.
#
# IMG_SIZE :
#     taille des images utilisées par le CNN.
#
# CROP_PAD :
#     paramètre utilisé plus tard lors de l'inférence.
from models.eye_cnn import EyeCNN, IMG_SIZE, CROP_PAD


# =====================================================================
#                         DONNÉES
# =====================================================================

def build_transforms(img_size: int):
    """
    Construit les transformations appliquées aux images.

    On crée deux pipelines différents :

        train_tf
            → utilisé pendant l'entraînement
            → contient de la data augmentation

        eval_tf
            → utilisé pour validation et test
            → aucune augmentation aléatoire

    Pourquoi ?

    Le modèle doit apprendre avec des images variées pendant
    l'entraînement.

    Mais lorsque nous mesurons ses performances, nous voulons utiliser
    des images propres et reproductibles.
    """

    # ---------------------------------------------------------------
    # TRANSFORMATIONS DU TRAIN
    # ---------------------------------------------------------------

    train_tf = transforms.Compose([

        # -----------------------------------------------------------
        # 1. CONVERSION EN NIVEAUX DE GRIS
        # -----------------------------------------------------------
        #
        # Une image RGB possède généralement 3 canaux :
        #
        #     R = Red
        #     G = Green
        #     B = Blue
        #
        # Ici nous convertissons l'image en un seul canal :
        #
        #     intensité lumineuse
        #
        # Pour déterminer si un œil est ouvert ou fermé,
        # la structure de l'œil est généralement plus importante
        # que sa couleur.
        transforms.Grayscale(),

        # -----------------------------------------------------------
        # 2. RANDOM RESIZED CROP
        # -----------------------------------------------------------
        #
        # On sélectionne aléatoirement une partie de l'image,
        # puis on la redimensionne en img_size × img_size.
        #
        # scale=(0.70, 1.0)
        #
        # signifie que le crop peut représenter entre 70 % et
        # 100 % de la surface originale.
        #
        # ratio=(0.85, 1.18)
        #
        # permet de varier légèrement le rapport largeur/hauteur.
        #
        # Pourquoi ?
        #
        # MediaPipe peut produire des crops légèrement différents
        # selon la position du visage et des landmarks.
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.70, 1.0),
            ratio=(0.85, 1.18)
        ),

        # -----------------------------------------------------------
        # 3. FLIP HORIZONTAL
        # -----------------------------------------------------------
        #
        # p=0.5 :
        #     50 % de chance de retourner horizontalement l'image.
        #
        # Cela est utile car un œil gauche peut être considéré
        # comme le miroir d'un œil droit.
        #
        # On augmente donc artificiellement la diversité des données.
        transforms.RandomHorizontalFlip(p=0.5),

        # -----------------------------------------------------------
        # 4. ROTATION
        # -----------------------------------------------------------
        #
        # L'image peut être tournée jusqu'à environ ±14 degrés.
        #
        # Cela simule une tête légèrement inclinée.
        transforms.RandomRotation(14),

        # -----------------------------------------------------------
        # 5. TRANSLATION
        # -----------------------------------------------------------
        #
        # L'image peut être légèrement déplacée horizontalement
        # et verticalement.
        #
        # translate=(0.08, 0.08)
        #
        # signifie que le déplacement peut aller jusqu'à environ
        # 8 % de la largeur/hauteur.
        #
        # Cela permet au CNN de ne pas supposer que l'œil est
        # toujours parfaitement centré.
        transforms.RandomAffine(
            0,
            translate=(0.08, 0.08)
        ),

        # -----------------------------------------------------------
        # 6. VARIATION DE LUMINOSITÉ ET CONTRASTE
        # -----------------------------------------------------------
        #
        # brightness=0.45
        #     → luminosité modifiée aléatoirement
        #
        # contrast=0.40
        #     → contraste modifié aléatoirement
        #
        # Cela simule différentes conditions d'éclairage :
        #
        #     ☀️ journée
        #     🌥️ lumière faible
        #     🌙 conduite de nuit
        transforms.ColorJitter(
            brightness=0.45,
            contrast=0.40
        ),

        # -----------------------------------------------------------
        # 7. FLOU
        # -----------------------------------------------------------
        #
        # RandomApply applique une transformation avec une certaine
        # probabilité.
        #
        # p=0.30
        #     → environ 30 % des images peuvent être floutées.
        #
        # Le flou simule :
        #
        #     - mouvement
        #     - vibration de caméra
        #     - mauvaise mise au point
        transforms.RandomApply(
            [
                transforms.GaussianBlur(
                    3,
                    sigma=(0.1, 1.6)
                )
            ],
            p=0.30
        ),

        # -----------------------------------------------------------
        # 8. CONVERSION EN TENSOR
        # -----------------------------------------------------------
        #
        # Le CNN PyTorch ne travaille pas directement avec une image
        # PIL classique.
        #
        # ToTensor convertit l'image en Tensor PyTorch.
        #
        # Exemple conceptuel :
        #
        #     Image
        #       ↓
        #     Tensor
        transforms.ToTensor(),

        # -----------------------------------------------------------
        # 9. NORMALISATION
        # -----------------------------------------------------------
        #
        # Pour un seul canal, on utilise :
        #
        #     mean = 0.5
        #     std  = 0.5
        #
        # La normalisation permet de placer les valeurs dans une
        # plage mieux adaptée à l'apprentissage du réseau.
        transforms.Normalize(
            [0.5],
            [0.5]
        ),

        # -----------------------------------------------------------
        # 10. RANDOM ERASING
        # -----------------------------------------------------------
        #
        # Avec p=0.25, une partie de l'image peut être effacée.
        #
        # Cela simule partiellement :
        #
        #     - monture de lunettes
        #     - cheveux
        #     - reflet
        #     - petite occlusion
        #
        # Le modèle doit donc apprendre à reconnaître l'état de
        # l'œil même si une petite partie est masquée.
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.12),
            value=0.0
        ),
    ])

    # ---------------------------------------------------------------
    # TRANSFORMATIONS VALIDATION / TEST
    # ---------------------------------------------------------------
    #
    # Ici aucune transformation aléatoire.
    #
    # On veut mesurer le modèle sur des images propres.
    eval_tf = transforms.Compose([

        # Même conversion en niveaux de gris.
        transforms.Grayscale(),

        # Toutes les images sont ramenées à la taille attendue
        # par le CNN.
        transforms.Resize(
            (img_size, img_size)
        ),

        # Conversion en Tensor.
        transforms.ToTensor(),

        # Même normalisation que pour le train.
        transforms.Normalize(
            [0.5],
            [0.5]
        ),
    ])

    # On retourne les deux pipelines.
    return train_tf, eval_tf


# =====================================================================
# NORMALISATION DES NOMS DE DOSSIERS
# =====================================================================

def _normalize_folders(
    data_dir: Path,
    open_dir: str,
    closed_dir: str
) -> None:
    """
    Renomme certains dossiers en noms standardisés.

    Exemple :

        Open_Eyes/
        Closed_Eyes/

    devient :

        open/
        closed/

    Cela permet à la suite du programme de toujours rechercher
    les mêmes noms de classes.
    """

    # On crée une correspondance :
    #
    # ancien nom → nouveau nom
    for src, dst in {
        open_dir: "open",
        closed_dir: "closed"
    }.items():

        # Chemin du dossier source.
        p_src = data_dir / src

        # Chemin du dossier destination.
        p_dst = data_dir / dst

        # On renomme uniquement si :
        #
        # 1. le dossier source existe
        # 2. le dossier destination n'existe pas
        # 3. les noms sont différents
        if (
            p_src.exists()
            and not p_dst.exists()
            and src != dst
        ):
            p_src.rename(p_dst)


# =====================================================================
# IDENTIFICATION DU SUJET
# =====================================================================

def subject_of(
    path: str,
    pattern: re.Pattern
) -> str | None:
    """
    Extrait l'identifiant du sujet depuis le nom du fichier.

    Exemple MRL :

        s0001_00123_0_0_0_0_0_01.png

    devient :

        s0001

    L'objectif est de savoir à quelle personne appartient chaque image.
    """

    # Path(path).name permet de récupérer uniquement le nom du fichier.
    #
    # Exemple :
    #
    # /dataset/open/s0001_00123.png
    #
    # devient :
    #
    # s0001_00123.png
    #
    # Ensuite pattern.search() cherche l'identifiant.
    m = pattern.search(
        Path(path).name
    )

    # Si un match est trouvé :
    #
    #     m.group(1)
    #
    # récupère le premier groupe capturé par l'expression régulière.
    #
    # Sinon :
    #
    #     None
    return m.group(1) if m else None


# =====================================================================
# SPLIT PAR SUJET
# =====================================================================

def group_split(
    samples: list[tuple[str, int]],
    pattern: re.Pattern,
    val_frac: float,
    test_frac: float,
    seed: int
):
    """
    Sépare le dataset par SUJET et non par image.

    Objectif :

        aucune personne ne doit apparaître simultanément
        dans train, validation et test.

    Exemple :

        Sujet A → toutes ses images → TRAIN

        Sujet B → toutes ses images → VALIDATION

        Sujet C → toutes ses images → TEST
    """

    # ---------------------------------------------------------------
    # Dictionnaire :
    #
    # sujet → indices des images appartenant à ce sujet
    #
    # Exemple :
    #
    # {
    #     "s0001": [0, 1, 2, 3],
    #     "s0002": [4, 5, 6],
    #     "s0003": [7, 8]
    # }
    # ---------------------------------------------------------------
    by_subject: dict[str, list[int]] = defaultdict(list)

    # Compteur d'images pour lesquelles aucun sujet n'a été trouvé.
    unknown = 0

    # enumerate() fournit :
    #
    # i    → index
    # path → chemin
    # _    → label
    #
    # Exemple :
    #
    # i = 25
    # path = ".../s0001_001.png"
    # _ = 0
    for i, (path, _) in enumerate(samples):

        # Recherche du sujet.
        s = subject_of(
            path,
            pattern
        )

        # Aucun sujet identifié.
        if s is None:

            # On incrémente le compteur.
            unknown += 1

            # Chaque image inconnue devient son propre groupe.
            #
            # Pourquoi ?
            #
            # Parce qu'on ne veut surtout pas mettre deux images
            # dont on ignore l'identité dans le même groupe par erreur.
            s = f"__unknown_{i}"

        # Ajout de l'index de l'image au groupe du sujet.
        by_subject[s].append(i)

    # ---------------------------------------------------------------
    # AVERTISSEMENT SI DES SUJETS SONT INCONNUS
    # ---------------------------------------------------------------

    if unknown:

        # Pourcentage d'images sans identifiant reconnu.
        pct = 100 * unknown / max(
            1,
            len(samples)
        )

        print(
            f"⚠ {unknown} images ({pct:.0f}%) sans identifiant "
            f"de sujet reconnu. Elles sont traitées comme des "
            f"sujets distincts — la protection contre la fuite "
            f"de données est partielle. Ajuste --subject-regex."
        )

    # ---------------------------------------------------------------
    # GÉNÉRATEUR ALÉATOIRE
    # ---------------------------------------------------------------

    # default_rng crée un générateur NumPy.
    #
    # Le seed permet de reproduire la même séparation.
    rng = np.random.default_rng(seed)

    # Liste des sujets.
    subjects = sorted(by_subject)

    # Mélange des sujets.
    rng.shuffle(subjects)

    # Nombre total d'images.
    n_total = len(samples)

    # Nombre approximatif d'images que l'on souhaite mettre
    # dans le test.
    want_test = test_frac * n_total

    # Nombre approximatif d'images que l'on souhaite mettre
    # dans la validation.
    want_val = val_frac * n_total

    # Listes qui contiendront les indices.
    test_idx = []
    val_idx = []
    train_idx = []

    # ---------------------------------------------------------------
    # RÉPARTITION DES SUJETS
    # ---------------------------------------------------------------

    # IMPORTANT :
    #
    # On parcourt les SUJETS.
    #
    # Pas les images individuellement.
    for s in subjects:

        # Toutes les images appartenant au sujet s.
        idxs = by_subject[s]

        # Tant qu'on n'a pas atteint le volume souhaité pour TEST :
        if len(test_idx) < want_test:

            # On ajoute TOUTES les images du sujet.
            test_idx += idxs

        # Sinon, si validation n'a pas encore atteint son objectif :
        elif len(val_idx) < want_val:

            # Toute la personne va en validation.
            val_idx += idxs

        # Sinon :
        else:

            # Toute la personne va dans train.
            train_idx += idxs

    # Nombre de sujets.
    n_sub = len(subjects)

    print(
        f"Découpe par sujet : {n_sub} sujets → "
        f"train {len(train_idx)} img · "
        f"val {len(val_idx)} img · "
        f"test {len(test_idx)} img"
    )

    # Si très peu de sujets :
    #
    # les résultats peuvent fortement varier selon les personnes
    # présentes dans le test.
    if n_sub < 8:
        print(
            "⚠ Moins de 8 sujets : la variance du score de test "
            "sera forte. Interprète-le comme un ordre de grandeur, "
            "pas comme une mesure fine."
        )

    # On retourne les trois ensembles d'indices.
    return (
        train_idx,
        val_idx,
        test_idx
    )


# =====================================================================
# UNE EPOCH D'ENTRAÎNEMENT / ÉVALUATION
# =====================================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train: bool
):
    """
    Effectue une passe complète sur un dataset.

    Si train=True :

        Forward
          ↓
        Loss
          ↓
        Backward
          ↓
        Optimizer step

    Si train=False :

        Forward
          ↓
        Loss
          ↓
        aucune modification des poids
    """

    # ---------------------------------------------------------------
    # MODE DU MODÈLE
    # ---------------------------------------------------------------

    # En entraînement :
    #
    #     model.train()
    #
    # En validation :
    #
    #     model.eval()
    #
    # Ces modes sont importants pour certains composants
    # comme Dropout ou BatchNorm.
    model.train() if train else model.eval()

    # Variables permettant de calculer les statistiques finales.
    total_loss = 0.0
    correct = 0
    total = 0

    # Active ou désactive le calcul des gradients.
    #
    # Train :
    #     gradients nécessaires
    #
    # Validation :
    #     gradients inutiles
    torch.set_grad_enabled(train)

    # ---------------------------------------------------------------
    # PARCOURS DES BATCHES
    # ---------------------------------------------------------------

    for images, labels in loader:

        # -----------------------------------------------------------
        # ENVOI VERS CPU OU GPU
        # -----------------------------------------------------------

        images = images.to(device)
        labels = labels.to(device)

        # -----------------------------------------------------------
        # 1. FORWARD PASS
        # -----------------------------------------------------------
        #
        # On donne les images au CNN.
        #
        # Exemple :
        #
        #     128 images
        #          ↓
        #         CNN
        #          ↓
        #     128 × 2 logits
        #
        # Chaque image obtient deux scores :
        #
        #     score open
        #     score closed
        logits = model(images)

        # -----------------------------------------------------------
        # CALCUL DE LA LOSS
        # -----------------------------------------------------------
        #
        # On compare les prédictions du réseau aux vraies labels.
        #
        # Exemple :
        #
        # Vrai :
        #     closed
        #
        # CNN :
        #     open   = 0.2
        #     closed = 0.8
        #
        # → erreur relativement faible.
        loss = criterion(
            logits,
            labels
        )

        # -----------------------------------------------------------
        # 2. BACKWARD PASS
        # -----------------------------------------------------------

        if train:

            # On supprime les gradients calculés au batch précédent.
            optimizer.zero_grad()

            # Backpropagation.
            #
            # PyTorch calcule automatiquement les gradients :
            #
            #     ∂Loss / ∂poids
            #
            # pour les paramètres du réseau.
            loss.backward()

            # -------------------------------------------------------
            # 3. MISE À JOUR DES POIDS
            # -------------------------------------------------------
            #
            # AdamW utilise les gradients pour modifier les poids.
            optimizer.step()

        # -----------------------------------------------------------
        # STATISTIQUES
        # -----------------------------------------------------------

        # loss.item() transforme le Tensor contenant la loss
        # en nombre Python.
        #
        # On multiplie par le nombre d'images pour obtenir une
        # somme pondérée permettant ensuite de calculer la moyenne.
        total_loss += (
            loss.item()
            * images.size(0)
        )

        # -----------------------------------------------------------
        # PRÉDICTION
        # -----------------------------------------------------------
        #
        # logits.argmax(1) sélectionne pour chaque image
        # l'indice du plus grand logit.
        #
        # Exemple :
        #
        #     [1.2, 3.8]
        #
        # max = 3.8
        # index = 1
        #
        # Si closed = 1 :
        #
        #     prédiction = closed
        correct += (
            (logits.argmax(1) == labels)
            .sum()
            .item()
        )

        # Nombre total d'images traitées.
        total += images.size(0)

    # On réactive les gradients pour la suite du programme.
    torch.set_grad_enabled(True)

    # ---------------------------------------------------------------
    # RÉSULTATS DE L'EPOCH
    # ---------------------------------------------------------------

    # Loss moyenne.
    average_loss = total_loss / max(
        1,
        total
    )

    # Accuracy :
    #
    # nombre de bonnes prédictions
    # -----------------------------
    # nombre total de prédictions
    accuracy = correct / max(
        1,
        total
    )

    return (
        average_loss,
        accuracy
    )


# =====================================================================
# COLLECTE DES LOGITS
# =====================================================================

@torch.no_grad()
def collect_logits(
    model,
    loader,
    device
):
    """
    Récupère les logits et les labels pour tout un dataset.

    On utilise cette fonction pour :

        - calibration
        - évaluation
    """

    # Mode évaluation.
    model.eval()

    # Liste des logits.
    L = []

    # Liste des labels.
    Y = []

    # Parcours des batches.
    for images, labels in loader:

        # Forward pass.
        logits = model(
            images.to(device)
        )

        # On ramène les logits sur CPU.
        L.append(
            logits.cpu()
        )

        # On conserve les labels.
        Y.append(labels)

    # Concatène tous les batches.
    #
    # Exemple :
    #
    # Batch 1 → 128 logits
    # Batch 2 → 128 logits
    # Batch 3 → 128 logits
    #
    # torch.cat()
    #     ↓
    # tous les logits ensemble
    return (
        torch.cat(L),
        torch.cat(Y)
    )


# =====================================================================
# CALIBRATION DE TEMPÉRATURE
# =====================================================================

def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """
    Apprend une température T sur la validation.

    Important :
        les poids du CNN ne sont PAS modifiés ici.

    On cherche simplement une valeur T telle que :

        logits_calibrés = logits / T

    produise des probabilités mieux calibrées.
    """

    # On travaille avec log(T) plutôt que T directement.
    #
    # Pourquoi ?
    #
    # Parce que :
    #
    #     exp(log_t) > 0
    #
    # donc T sera toujours positive.
    log_t = torch.zeros(
        1,
        requires_grad=True
    )

    # LBFGS est un optimiseur utilisé ici pour apprendre
    # la température.
    opt = torch.optim.LBFGS(
        [log_t],
        lr=0.05,
        max_iter=120
    )

    # Negative Log Likelihood / Cross Entropy.
    nll = nn.CrossEntropyLoss()

    # Fonction appelée par LBFGS.
    def closure():

        # Supprime les anciens gradients.
        opt.zero_grad()

        # Température :
        #
        #     T = exp(log_t)
        #
        # Puis :
        #
        #     logits / T
        loss = nll(
            logits / log_t.exp(),
            labels
        )

        # Calcule le gradient par rapport à log_t.
        loss.backward()

        return loss

    # Optimisation de la température.
    opt.step(closure)

    # Conversion en float Python.
    return float(
        log_t.exp().item()
    )


# =====================================================================
# CHOIX DU SEUIL
# =====================================================================

def pick_threshold(
    p_closed: np.ndarray,
    is_closed: np.ndarray,
    target_recall: float
) -> float:
    """
    Recherche le seuil permettant d'atteindre le rappel cible
    sur la classe "closed".

    Exemple :

        target_recall = 0.98

    signifie :

        recall closed >= 98 %

    On recherche le seuil le PLUS HAUT qui respecte cette contrainte.

    Pourquoi le plus haut ?

    Parce qu'à rappel équivalent, un seuil plus haut tend à limiter
    davantage les prédictions "closed" inutiles.
    """

    # On récupère les probabilités distinctes comme candidats
    # de seuil.
    #
    # np.round(..., 4)
    #     → arrondit à 4 décimales
    #
    # np.unique()
    #     → supprime les doublons.
    order = np.unique(
        np.round(
            p_closed,
            4
        )
    )

    # Aucun seuil trouvé pour l'instant.
    best = None

    # Test de chaque seuil.
    for thr in order:

        # Une image est considérée comme "closed" si :
        #
        #     P(closed) >= threshold
        pred = p_closed >= thr

        # True Positive :
        #
        # prédiction = closed
        # ET
        # vérité = closed
        tp = int(
            (pred & is_closed).sum()
        )

        # Rappel :
        #
        #             TP
        # Recall = -----------
        #          TP + FN
        recall = tp / max(
            1,
            int(is_closed.sum())
        )

        # Si le rappel cible est atteint :
        if recall >= target_recall:

            # On garde ce seuil.
            #
            # Comme les seuils sont parcourus dans l'ordre croissant,
            # chaque nouveau seuil valide est plus grand.
            best = float(thr)

    # Aucun seuil n'a permis d'atteindre la cible.
    if best is None:

        print(
            f"⚠ Rappel cible {target_recall:.2f} inatteignable "
            f"sur la validation. Seuil ramené à 0.10 "
            f"(rappel maximal)."
        )

        return 0.10

    # Retourne le seuil choisi.
    return best


# =====================================================================
# ÉVALUATION
# =====================================================================

def evaluate(
    p_closed: np.ndarray,
    is_closed: np.ndarray,
    threshold: float
) -> dict:
    """
    Calcule les métriques finales.

    IMPORTANT :

        "closed" = CLASSE POSITIVE

    Donc :

        Positive = œil fermé
        Negative = œil ouvert

    Matrice :

                      PRÉDIT
                  closed   open

        RÉEL closed   TP      FN
        RÉEL open     FP      TN
    """

    # Import local des métriques sklearn.
    from sklearn.metrics import (
        roc_auc_score,
        confusion_matrix,
        average_precision_score
    )

    # ---------------------------------------------------------------
    # TRANSFORMATION PROBABILITÉ → DÉCISION
    # ---------------------------------------------------------------

    # Si :
    #
    #     P(closed) >= threshold
    #
    # alors :
    #
    #     closed
    #
    # sinon :
    #
    #     open
    pred = p_closed >= threshold

    # ---------------------------------------------------------------
    # MATRICE DE CONFUSION
    # ---------------------------------------------------------------

    # True Positive :
    #
    # réellement fermé
    # ET
    # prédit fermé
    tp = int(
        (pred & is_closed).sum()
    )

    # False Positive :
    #
    # réellement ouvert
    # MAIS
    # prédit fermé
    fp = int(
        (pred & ~is_closed).sum()
    )

    # False Negative :
    #
    # réellement fermé
    # MAIS
    # prédit ouvert
    #
    # C'est particulièrement important dans ce contexte.
    fn = int(
        (~pred & is_closed).sum()
    )

    # True Negative :
    #
    # réellement ouvert
    # ET
    # prédit ouvert
    tn = int(
        (~pred & ~is_closed).sum()
    )

    # ---------------------------------------------------------------
    # PRECISION
    # ---------------------------------------------------------------
    #
    # Parmi les images que le modèle a déclarées "closed",
    # combien étaient réellement closed ?
    prec = tp / max(
        1,
        tp + fp
    )

    # ---------------------------------------------------------------
    # RECALL
    # ---------------------------------------------------------------
    #
    # Parmi tous les yeux réellement fermés,
    # combien ont été détectés ?
    rec = tp / max(
        1,
        tp + fn
    )

    # ---------------------------------------------------------------
    # RÉSULTATS
    # ---------------------------------------------------------------

    return {

        # Documentation explicite de la classe positive.
        "positive_class": "closed",

        # Seuil utilisé.
        "threshold": float(threshold),

        # -----------------------------------------------------------
        # ACCURACY
        # -----------------------------------------------------------
        #
        #             TP + TN
        # Accuracy = -----------
        #                N
        "accuracy": float(
            (tp + tn)
            / max(
                1,
                len(is_closed)
            )
        ),

        # -----------------------------------------------------------
        # AUROC
        # -----------------------------------------------------------
        #
        # Mesure la capacité du modèle à distinguer les deux classes
        # sur différents seuils.
        "auroc": float(
            roc_auc_score(
                is_closed,
                p_closed
            )
        ),

        # Average Precision.
        "average_precision": float(
            average_precision_score(
                is_closed,
                p_closed
            )
        ),

        # Precision sur la classe closed.
        "precision_closed": float(
            prec
        ),

        # Recall sur la classe closed.
        "recall_closed": float(
            rec
        ),

        # -----------------------------------------------------------
        # F1 SCORE
        # -----------------------------------------------------------
        #
        # F1 combine Precision et Recall.
        "f1_closed": float(
            2 * prec * rec
            / max(
                1e-9,
                prec + rec
            )
        ),

        # -----------------------------------------------------------
        # MISS RATE
        # -----------------------------------------------------------
        #
        # Proportion des yeux fermés qui ont été ratés.
        #
        # miss rate = FN / (TP + FN)
        #
        # C'est également :
        #
        # miss rate = 1 - recall
        "miss_rate_closed": float(
            fn
            / max(
                1,
                tp + fn
            )
        ),

        # -----------------------------------------------------------
        # FALSE ALARM RATE
        # -----------------------------------------------------------
        #
        # Parmi les yeux réellement ouverts,
        # combien ont été déclarés fermés ?
        "false_alarm_rate": float(
            fp
            / max(
                1,
                fp + tn
            )
        ),

        # -----------------------------------------------------------
        # MATRICE DE CONFUSION
        # -----------------------------------------------------------
        #
        # lignes = réalité
        # colonnes = prédiction
        #
        # ordre :
        #
        #     [closed, open]
        "confusion": confusion_matrix(
            is_closed.astype(int),
            pred.astype(int),
            labels=[1, 0]
        ).tolist(),
    }


# =====================================================================
#                         FIGURES
# =====================================================================

def plot_curves(
    history,
    out: Path
):
    """
    Trace les courbes d'apprentissage :

        - train loss
        - validation loss
        - train accuracy
        - validation accuracy

    Elles permettent notamment d'observer l'évolution du modèle
    pendant l'entraînement et de détecter un éventuel overfitting.
    """

    import matplotlib

    # Utilisation d'un backend sans interface graphique.
    # Pratique lorsqu'on exécute le script sur un serveur.
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    # Deux graphiques côte à côte.
    fig, (a1, a2) = plt.subplots(
        1,
        2,
        figsize=(11, 4)
    )

    # ---------------------------------------------------------------
    # GRAPHIQUE LOSS
    # ---------------------------------------------------------------

    a1.plot(
        history["train_loss"],
        label="train"
    )

    a1.plot(
        history["val_loss"],
        label="val"
    )

    a1.set_title("Loss")
    a1.set_xlabel("epoch")
    a1.legend()

    # ---------------------------------------------------------------
    # GRAPHIQUE ACCURACY
    # ---------------------------------------------------------------

    a2.plot(
        history["train_acc"],
        label="train"
    )

    a2.plot(
        history["val_acc"],
        label="val"
    )

    a2.set_title("Accuracy")
    a2.set_xlabel("epoch")
    a2.legend()

    # Ajustement automatique de la disposition.
    fig.tight_layout()

    # Sauvegarde du graphique.
    fig.savefig(
        out,
        dpi=130
    )

    # Libère la mémoire.
    plt.close(fig)


# =====================================================================
# MATRICE DE CONFUSION
# =====================================================================

def plot_confusion(
    cm,
    out: Path
):
    """
    Génère une représentation graphique de la matrice de confusion.
    """

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    # Conversion en tableau NumPy.
    cm = np.array(cm)

    # Création de la figure.
    fig, ax = plt.subplots(
        figsize=(4.4, 4.2)
    )

    # Affichage de la matrice.
    ax.imshow(
        cm,
        cmap="Blues"
    )

    # Positions des classes.
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    # Labels :
    #
    # colonne = prédiction
    ax.set_xticklabels(
        ["fermé", "ouvert"]
    )

    # ligne = réalité
    ax.set_yticklabels(
        ["fermé", "ouvert"]
    )

    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")

    # ---------------------------------------------------------------
    # ÉCRIT LES VALEURS DANS LES CASES
    # ---------------------------------------------------------------

    for i in range(2):

        for j in range(2):

            # On écrit le nombre dans chaque case.
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=14,

                # Texte blanc si la case est sombre,
                # noir sinon.
                color=(
                    "white"
                    if cm[i, j] > cm.max() / 2
                    else "black"
                )
            )

    ax.set_title(
        "Matrice de confusion "
        "(test, sujets inédits)"
    )

    fig.tight_layout()

    fig.savefig(
        out,
        dpi=130
    )

    plt.close(fig)


# =====================================================================
# COURBE ROC
# =====================================================================

def plot_roc(
    p_closed,
    is_closed,
    threshold,
    out: Path
):
    """
    Génère la courbe ROC.

    La courbe permet de visualiser le compromis entre :

        - rappel
        - taux de fausses alarmes

    Le seuil choisi est également affiché sur la courbe.
    """

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from sklearn.metrics import roc_curve

    # Calcul de la courbe ROC.
    #
    # fpr :
    #     False Positive Rate
    #
    # tpr :
    #     True Positive Rate = Recall
    #
    # thr :
    #     seuils correspondants
    fpr, tpr, thr = roc_curve(
        is_closed,
        p_closed
    )

    # Création de la figure.
    fig, ax = plt.subplots(
        figsize=(4.6, 4.4)
    )

    # Courbe ROC.
    ax.plot(
        fpr,
        tpr,
        lw=2
    )

    # Ligne diagonale représentant approximativement
    # un classifieur sans capacité de séparation.
    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        lw=1,
        color="#999"
    )

    # Recherche du point de la courbe dont le seuil est
    # le plus proche du seuil retenu.
    k = int(
        np.argmin(
            np.abs(
                thr - threshold
            )
        )
    )

    # Affiche le seuil retenu.
    ax.scatter(
        [fpr[k]],
        [tpr[k]],
        s=60,
        zorder=3,
        color="#DC4B39"
    )

    # Texte associé au point.
    ax.annotate(
        f"seuil retenu {threshold:.2f}\n"
        f"rappel {tpr[k]:.3f}",
        (fpr[k], tpr[k]),
        textcoords="offset points",
        xytext=(10, -24),
        fontsize=9
    )

    ax.set_xlabel(
        "Taux de fausses alarmes"
    )

    ax.set_ylabel(
        "Rappel sur œil fermé"
    )

    ax.set_title(
        "ROC — point de fonctionnement choisi"
    )

    fig.tight_layout()

    fig.savefig(
        out,
        dpi=130
    )

    plt.close(fig)


# =====================================================================
#                              MAIN
# =====================================================================

def main():
    """
    Fonction principale.

    C'est elle qui orchestre tout le pipeline :

        paramètres
          ↓
        dataset
          ↓
        split
          ↓
        DataLoader
          ↓
        CNN
          ↓
        entraînement
          ↓
        calibration
          ↓
        seuil
          ↓
        test
          ↓
        sauvegarde
    """

    # =================================================================
    # ARGUMENTS DU PROGRAMME
    # =================================================================

    # Création du parser.
    ap = argparse.ArgumentParser()

    # ---------------------------------------------------------------
    # CHEMIN DU DATASET
    # ---------------------------------------------------------------

    ap.add_argument(
        "--data",
        default="train/data"
    )

    # ---------------------------------------------------------------
    # NOMBRE D'EPOCHS
    # ---------------------------------------------------------------

    ap.add_argument(
        "--epochs",
        type=int,
        default=25
    )

    # ---------------------------------------------------------------
    # TAILLE DU BATCH
    # ---------------------------------------------------------------

    # Nombre d'images traitées simultanément.
    ap.add_argument(
        "--batch",
        type=int,
        default=128
    )

    # ---------------------------------------------------------------
    # LEARNING RATE
    # ---------------------------------------------------------------

    # Contrôle la taille des mises à jour des poids.
    ap.add_argument(
        "--lr",
        type=float,
        default=1.5e-3
    )

    # ---------------------------------------------------------------
    # PATIENCE
    # ---------------------------------------------------------------

    # Nombre d'epochs sans amélioration avant early stopping.
    ap.add_argument(
        "--patience",
        type=int,
        default=6
    )

    # ---------------------------------------------------------------
    # TAILLE DES IMAGES
    # ---------------------------------------------------------------

    ap.add_argument(
        "--img-size",
        type=int,
        default=IMG_SIZE
    )

    # ---------------------------------------------------------------
    # LARGEUR DU CNN
    # ---------------------------------------------------------------

    # Nombre de canaux du premier bloc convolutionnel.
    ap.add_argument(
        "--width",
        type=int,
        default=32,
        help="largeur du 1er bloc conv"
    )

    # ---------------------------------------------------------------
    # RAPPEL CIBLE
    # ---------------------------------------------------------------

    # On souhaite par défaut atteindre 98 % de rappel
    # sur les yeux fermés.
    ap.add_argument(
        "--target-recall",
        type=float,
        default=0.98,
        help=(
            "rappel visé sur la classe 'fermé' "
            "pour choisir le seuil"
        )
    )

    # ---------------------------------------------------------------
    # REGEX POUR IDENTIFIER LE SUJET
    # ---------------------------------------------------------------

    # Expression régulière par défaut.
    #
    # Elle permet par exemple de reconnaître :
    #
    #     s0001_...
    #
    # comme sujet :
    #
    #     s0001
    ap.add_argument(
        "--subject-regex",
        default=r"^([A-Za-z]?\d+)_",
        help=(
            "capture l'identifiant du sujet "
            "dans le nom de fichier"
        )
    )

    # ---------------------------------------------------------------
    # OPTION POUR UN SPLIT ALÉATOIRE
    # ---------------------------------------------------------------

    # Cette option permet de désactiver le split par sujet.
    #
    # Elle est déconseillée car elle peut produire un score
    # artificiellement optimiste si plusieurs frames proches
    # d'une même personne se retrouvent dans train et test.
    ap.add_argument(
        "--random-split",
        action="store_true",
        help=(
            "désactive la découpe par sujet "
            "(déconseillé, gonfle le score)"
        )
    )

    # ---------------------------------------------------------------
    # SEED
    # ---------------------------------------------------------------

    # Permet de rendre certaines opérations aléatoires reproductibles.
    ap.add_argument(
        "--seed",
        type=int,
        default=42
    )

    # ---------------------------------------------------------------
    # NOMS DES DOSSIERS
    # ---------------------------------------------------------------

    ap.add_argument(
        "--open-dir",
        default="open"
    )

    ap.add_argument(
        "--closed-dir",
        default="closed"
    )

    # Récupération des arguments.
    args = ap.parse_args()


    # =================================================================
    # REPRODUCTIBILITÉ
    # =================================================================

    # Fixe la seed PyTorch.
    #
    # Cela permet d'obtenir des résultats plus reproductibles.
    torch.manual_seed(
        args.seed
    )


    # =================================================================
    # CPU OU GPU ?
    # =================================================================

    # Si CUDA est disponible :
    #
    #     utilisation du GPU
    #
    # Sinon :
    #
    #     utilisation du CPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device : {device}"
    )


    # =================================================================
    # PRÉPARATION DU DATASET
    # =================================================================

    # Conversion du chemin du dataset en objet Path.
    data_dir = Path(
        args.data
    )

    # Normalisation des noms de dossiers.
    _normalize_folders(
        data_dir,
        args.open_dir,
        args.closed_dir
    )

    # Construction des deux pipelines de transformations :
    #
    # train_tf :
    #     avec augmentation
    #
    # eval_tf :
    #     sans augmentation
    train_tf, eval_tf = build_transforms(
        args.img_size
    )


    # =================================================================
    # CRÉATION DES DATASETS
    # =================================================================

    # Dataset utilisé pour TRAIN.
    #
    # Il utilise les transformations augmentées.
    ds_train_view = ImageFolder(
        str(data_dir),
        transform=train_tf
    )

    # Dataset utilisé pour validation/test.
    #
    # Il utilise les transformations propres.
    ds_eval_view = ImageFolder(
        str(data_dir),
        transform=eval_tf
    )


    # =================================================================
    # MAPPING CLASSE → INDEX
    # =================================================================

    # ImageFolder construit automatiquement un dictionnaire.
    #
    # Exemple :
    #
    #     {
    #         "closed": 0,
    #         "open": 1
    #     }
    #
    # ou inversement selon les noms.
    class_to_idx = ds_eval_view.class_to_idx

    # On récupère l'index correspondant à "closed".
    closed_idx = class_to_idx.get(
        "closed"
    )

    # Si le dossier closed n'existe pas :
    if closed_idx is None:

        raise SystemExit(
            f"Dossier 'closed' introuvable. "
            f"Classes vues : {class_to_idx}"
        )

    print(
        f"Classes : {class_to_idx} "
        f"(index de 'closed' = {closed_idx})"
    )


    # =================================================================
    # DÉCOUPE TRAIN / VAL / TEST
    # =================================================================

    if args.random_split:

        # -------------------------------------------------------------
        # SPLIT ALÉATOIRE
        # -------------------------------------------------------------
        #
        # Cette méthode sépare les images individuellement.
        #
        # ATTENTION :
        #
        # des frames très proches du même œil peuvent alors
        # être présentes dans train ET test.
        print(
            "⚠ Découpe aléatoire demandée : des frames voisines "
            "du même œil peuvent se retrouver dans train ET test. "
            "Le score sera optimiste."
        )

        # Nombre total d'images.
        n = len(
            ds_eval_view
        )

        # Générateur aléatoire.
        rng = np.random.default_rng(
            args.seed
        )

        # Mélange des indices.
        perm = rng.permutation(n)

        # 15 % pour test.
        n_test = int(
            0.15 * n
        )

        # 15 % pour validation.
        n_val = int(
            0.15 * n
        )

        # Premiers indices → test.
        test_idx = perm[
            :n_test
        ].tolist()

        # Indices suivants → validation.
        val_idx = perm[
            n_test:n_test + n_val
        ].tolist()

        # Le reste → train.
        train_idx = perm[
            n_test + n_val:
        ].tolist()

    else:

        # -------------------------------------------------------------
        # SPLIT PAR SUJET
        # -------------------------------------------------------------

        # Compilation de l'expression régulière.
        pattern = re.compile(
            args.subject_regex
        )

        # Séparation par sujet.
        train_idx, val_idx, test_idx = group_split(
            ds_eval_view.samples,
            pattern,
            0.15,
            0.15,
            args.seed
        )


    # =================================================================
    # CRÉATION DES SUBSETS
    # =================================================================

    # Train :
    #
    # on utilise ds_train_view → augmentation active.
    train_set = Subset(
        ds_train_view,
        train_idx
    )

    # Validation :
    #
    # ds_eval_view → pas d'augmentation.
    val_set = Subset(
        ds_eval_view,
        val_idx
    )

    # Test :
    #
    # ds_eval_view → pas d'augmentation.
    test_set = Subset(
        ds_eval_view,
        test_idx
    )


    # =================================================================
    # DATALOADERS
    # =================================================================

    # ---------------------------------------------------------------
    # TRAIN LOADER
    # ---------------------------------------------------------------
    #
    # batch_size=128 :
    #
    # le CNN reçoit 128 images à la fois.
    #
    # shuffle=True :
    #
    # mélange les données d'entraînement.
    #
    # num_workers=6 :
    #
    # utilise plusieurs processus pour charger les images.
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=6
    )

    # Validation.
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        num_workers=6
    )

    # Test.
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch,
        num_workers=6
    )


    # =================================================================
    # CRÉATION DU CNN
    # =================================================================

    # Création de l'architecture.
    #
    # EyeCNN est défini dans :
    #
    #     models/eye_cnn.py
    #
    # img_size :
    #     taille d'entrée.
    #
    # width :
    #     largeur du premier bloc convolutionnel.
    #
    # .to(device) :
    #     envoie le modèle sur CPU ou GPU.
    model = EyeCNN(
        img_size=args.img_size,
        width=args.width
    ).to(device)

    print(
        f"Modèle : {model.n_params():,} paramètres, "
        f"entrée {args.img_size}×{args.img_size}"
    )


    # =================================================================
    # GESTION DU DÉSÉQUILIBRE DES CLASSES
    # =================================================================

    # Compte combien d'images de chaque classe sont présentes
    # dans TRAIN.
    #
    # Exemple :
    #
    #     open   = 9000
    #     closed = 3000
    #
    # Si le dataset est déséquilibré, le modèle pourrait être tenté
    # de privilégier la classe majoritaire.
    counts = np.bincount(
        [
            ds_eval_view.samples[i][1]
            for i in train_idx
        ],
        minlength=2
    )

    # ---------------------------------------------------------------
    # CALCUL DES POIDS DE CLASSES
    # ---------------------------------------------------------------
    #
    # La classe minoritaire reçoit davantage de poids dans la loss.
    #
    # Ainsi, une erreur sur une classe rare coûte davantage au réseau.
    weights = torch.tensor(
        (
            counts.sum()
            / np.maximum(counts, 1)
        )
        / 2.0,
        dtype=torch.float32
    )

    print(
        f"Répartition train : "
        f"{dict(zip(ds_eval_view.classes, counts.tolist()))} "
        f"→ poids {weights.tolist()}"
    )


    # =================================================================
    # FONCTION DE LOSS
    # =================================================================

    # CrossEntropyLoss mesure l'écart entre :
    #
    #     prédiction du CNN
    #
    # et :
    #
    #     vraie classe
    #
    # weight :
    #     corrige le déséquilibre entre classes.
    #
    # label_smoothing=0.03 :
    #     évite des cibles trop rigides et peut limiter
    #     une confiance excessive.
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device),
        label_smoothing=0.03
    )


    # =================================================================
    # OPTIMIZER
    # =================================================================

    # AdamW est responsable de la mise à jour des poids.
    #
    # learning rate :
    #     taille des mises à jour
    #
    # weight_decay :
    #     régularisation pour limiter certains phénomènes
    #     d'overfitting.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )


    # =================================================================
    # LEARNING RATE SCHEDULER
    # =================================================================

    # Le learning rate évolue progressivement pendant l'entraînement.
    #
    # CosineAnnealing fait diminuer le learning rate selon
    # une fonction de type cosinus.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )


    # =================================================================
    # DOSSIER DE SORTIE
    # =================================================================

    # Les modèles et rapports seront sauvegardés ici :
    #
    #     train/models/
    out_dir = Path(
        "train/models"
    )

    # Création du dossier s'il n'existe pas.
    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # =================================================================
    # HISTORIQUE DE L'ENTRAÎNEMENT
    # =================================================================

    # On conserve les valeurs de :
    #
    #     train loss
    #     validation loss
    #     train accuracy
    #     validation accuracy
    #
    # à chaque epoch.
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }


    # =================================================================
    # EARLY STOPPING
    # =================================================================

    # Meilleure validation observée.
    best_val = 1e9

    # Nombre d'epochs restantes avant arrêt.
    patience_left = args.patience

    # Contiendra les poids du meilleur modèle.
    best_state = None


    # =================================================================
    # BOUCLE D'ENTRAÎNEMENT
    # =================================================================

    # Exemple :
    #
    # args.epochs = 25
    #
    # On fera :
    #
    # epoch 1
    # epoch 2
    # ...
    # epoch 25
    for epoch in range(
        1,
        args.epochs + 1
    ):

        # -------------------------------------------------------------
        # TRAIN
        # -------------------------------------------------------------
        #
        # Ici :
        #
        # train=True
        #
        # donc :
        #
        #     forward
        #     loss
        #     backward
        #     optimizer.step()
        tl, ta = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            True
        )

        # -------------------------------------------------------------
        # VALIDATION
        # -------------------------------------------------------------
        #
        # train=False
        #
        # donc aucun poids n'est modifié.
        vl, va = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            False
        )

        # Mise à jour du learning rate.
        scheduler.step()


        # -------------------------------------------------------------
        # SAUVEGARDE DE L'HISTORIQUE
        # -------------------------------------------------------------

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)

        history["train_acc"].append(ta)
        history["val_acc"].append(va)


        # -------------------------------------------------------------
        # AFFICHAGE
        # -------------------------------------------------------------

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {tl:.3f} acc {ta:.3f} | "
            f"val loss {vl:.3f} acc {va:.3f}"
        )


        # -------------------------------------------------------------
        # MEILLEUR MODÈLE ?
        # -------------------------------------------------------------

        # Si la validation s'améliore suffisamment :
        if vl < best_val - 1e-4:

            # Nouvelle meilleure loss de validation.
            best_val = vl

            # Réinitialisation de la patience.
            patience_left = args.patience

            # Sauvegarde d'une copie des poids.
            #
            # .cpu()
            #     → copie sur CPU
            #
            # .clone()
            #     → copie indépendante
            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

        else:

            # Pas d'amélioration.
            patience_left -= 1

            # Si aucune amélioration pendant assez longtemps :
            if patience_left == 0:

                print(
                    f"Early stopping à l'epoch {epoch}."
                )

                # On arrête l'entraînement.
                break


    # =================================================================
    # RESTAURATION DU MEILLEUR MODÈLE
    # =================================================================

    # Si un meilleur état a été sauvegardé :
    if best_state:

        # On recharge ses poids.
        model.load_state_dict(
            best_state
        )

    # On remet le modèle sur CPU/GPU.
    model.to(device)


    # =================================================================
    # CALIBRATION + CHOIX DU SEUIL
    # =================================================================
    #
    # IMPORTANT :
    #
    # Ces deux étapes utilisent UNIQUEMENT VALIDATION.
    #
    # Le TEST reste complètement séparé.
    # =================================================================

    # Récupération des logits et labels de validation.
    val_logits, val_labels = collect_logits(
        model,
        val_loader,
        device
    )

    # ---------------------------------------------------------------
    # CALIBRATION DE TEMPÉRATURE
    # ---------------------------------------------------------------

    temperature = fit_temperature(
        val_logits,
        val_labels
    )

    # ---------------------------------------------------------------
    # PROBABILITÉ "CLOSED"
    # ---------------------------------------------------------------
    #
    # 1. logits / temperature
    #
    # 2. softmax
    #
    # 3. [:, closed_idx]
    #
    # On récupère la probabilité correspondant spécifiquement
    # à la classe "closed".
    val_p = torch.softmax(
        val_logits / temperature,
        dim=1
    )[:, closed_idx].numpy()

    # ---------------------------------------------------------------
    # VÉRITÉ TERRAIN
    # ---------------------------------------------------------------
    #
    # True si le label réel est "closed".
    val_closed = (
        val_labels.numpy()
        == closed_idx
    )

    # ---------------------------------------------------------------
    # CHOIX DU SEUIL
    # ---------------------------------------------------------------

    threshold = pick_threshold(
        val_p,
        val_closed,
        args.target_recall
    )

    print(
        f"\nCalibration : "
        f"température {temperature:.3f} · "
        f"seuil retenu {threshold:.3f} "
        f"(cible rappel 'fermé' "
        f"{args.target_recall:.2f})"
    )


    # =================================================================
    # ÉVALUATION FINALE SUR TEST
    # =================================================================
    #
    # Ici nous utilisons des sujets qui n'ont pas été utilisés
    # pour entraîner le modèle.
    # =================================================================

    # Récupération des logits du test.
    test_logits, test_labels = collect_logits(
        model,
        test_loader,
        device
    )

    # Conversion des logits en probabilités.
    #
    # On utilise la température apprise sur VALIDATION.
    test_p = torch.softmax(
        test_logits / temperature,
        dim=1
    )[:, closed_idx].numpy()

    # Vérité terrain :
    #
    # True = œil réellement fermé
    test_closed = (
        test_labels.numpy()
        == closed_idx
    )


    # =================================================================
    # CALCUL DES MÉTRIQUES
    # =================================================================

    metrics = evaluate(
        test_p,
        test_closed,
        threshold
    )


    # =================================================================
    # AJOUT DE MÉTADONNÉES
    # =================================================================

    # On ajoute des informations utiles au rapport.
    metrics.update({

        # Température de calibration.
        "temperature": temperature,

        # Mapping des classes.
        "class_to_idx": class_to_idx,

        # Nombre de paramètres du CNN.
        "n_params": model.n_params(),

        # Taille des images.
        "img_size": args.img_size,

        # Largeur du réseau.
        "width": args.width,

        # Type de split utilisé.
        "split": (
            "random"
            if args.random_split
            else "par sujet"
        ),

        # Nombre d'images dans chaque ensemble.
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx)
    })


    # =================================================================
    # AFFICHAGE DES RÉSULTATS
    # =================================================================

    print(
        "\n═════ TEST (sujets inédits) ═════"
    )

    # Accuracy globale.
    print(
        f"Accuracy          : "
        f"{metrics['accuracy']:.4f}"
    )

    # AUROC.
    print(
        f"AUROC             : "
        f"{metrics['auroc']:.4f}"
    )

    # Precision sur closed.
    print(
        f"Précision (fermé) : "
        f"{metrics['precision_closed']:.4f}"
    )

    # Recall sur closed.
    print(
        f"Rappel (fermé)    : "
        f"{metrics['recall_closed']:.4f}"
    )

    # Taux d'yeux fermés ratés.
    print(
        f"Yeux fermés ratés : "
        f"{metrics['miss_rate_closed'] * 100:.2f}% "
        f"← le chiffre à surveiller"
    )

    # Taux de fausses alarmes.
    print(
        f"Fausses alarmes   : "
        f"{metrics['false_alarm_rate'] * 100:.2f}%"
    )

    # Matrice de confusion.
    print(
        f"Confusion [fermé, ouvert] : "
        f"{metrics['confusion']}"
    )


    # =================================================================
    # SAUVEGARDE DU MODÈLE
    # =================================================================

    # torch.save() sauvegarde plusieurs éléments nécessaires
    # pour refaire l'inférence plus tard.
    #
    # Ce n'est donc pas seulement "les poids".
    #
    # On sauvegarde aussi :
    #
    #     - mapping des classes
    #     - taille d'image
    #     - architecture
    #     - seuil
    #     - température
    #     - métriques
    torch.save(
        {
            "state_dict": model.state_dict(),

            "class_to_idx": class_to_idx,

            "img_size": args.img_size,

            "width": args.width,

            "threshold": threshold,

            "temperature": temperature,

            "crop_pad": CROP_PAD,

            "metrics": metrics
        },

        out_dir / "eye_cnn.pt"
    )


    # =================================================================
    # SAUVEGARDE DES MÉTRIQUES
    # =================================================================

    # Conversion du dictionnaire Python en fichier JSON.
    (
        out_dir / "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2
        ),
        encoding="utf-8"
    )


    # =================================================================
    # SAUVEGARDE DES GRAPHIQUES
    # =================================================================

    # Courbes train / validation.
    plot_curves(
        history,
        out_dir / "training_curves.png"
    )

    # Matrice de confusion.
    plot_confusion(
        metrics["confusion"],
        out_dir / "confusion.png"
    )

    # Courbe ROC.
    plot_roc(
        test_p,
        test_closed,
        threshold,
        out_dir / "roc.png"
    )


    # =================================================================
    # FIN
    # =================================================================

    print(
        f"\n✅ Modèle et rapports dans "
        f"{out_dir}/"
    )

    print(
        "   Le seuil et la température sont dans le checkpoint : "
        "l'inférence les reprend automatiquement."
    )


# =====================================================================
# POINT D'ENTRÉE DU PROGRAMME
# =====================================================================

# Cette condition signifie :
#
# "Si ce fichier est exécuté directement, appelle main()."
#
# Exemple :
#
#     python train/train_eye_cnn.py
#
# → main() sera exécutée.
#
# Mais si le fichier est importé par un autre script :
#
#     import train_eye_cnn
#
# → main() ne sera pas exécutée automatiquement.
if __name__ == "__main__":
    main()