"""Wrapper landmarks visage — DOUBLE BACKEND MediaPipe.

Les versions récentes de MediaPipe (Python 3.12/3.13, Windows notamment) ont
supprimé l'ancienne API `mp.solutions`.

Ce module gère donc deux versions possibles de MediaPipe :

1. L'ancienne API `mp.solutions.face_mesh`
2. La nouvelle API `mediapipe.tasks.python.vision.FaceLandmarker`

Dans les deux cas, l'objectif est le même :
    image BGR
        ↓
    détection des landmarks du visage
        ↓
    tableau NumPy contenant les coordonnées (x, y, z)

Interface publique :
    FaceLandmarker(...).process(frame_bgr)

Retourne :
    np.ndarray de forme (N, 3)
ou :
    None si aucun visage n'est détecté.
"""

# Permet d'utiliser plus souplement les annotations de types Python.
from __future__ import annotations
# Module Python permettant de télécharger un fichier depuis une URL.
import urllib.request
# Path permet de manipuler les chemins de fichiers de manière propre et portable.
from pathlib import Path
# NumPy sera utilisé pour stocker les coordonnées des landmarks.
import numpy as np


# On essaye d'importer la bibliothèque MediaPipe.
try:
    import mediapipe as mp

# Si MediaPipe n'est pas installé, Python déclenche une ImportError.
except ImportError as e:

    # On renvoie une erreur plus claire à l'utilisateur.
    raise ImportError(
        "MediaPipe manquant : pip install mediapipe"
    ) from e

# ============================================================
# INDICES DES LANDMARKS
# ============================================================

# MediaPipe associe un numéro précis à chaque point du visage.
# Ces indices correspondent ici à plusieurs points de l'œil gauche.
LEFT_EYE = [
    33,   # Coin extérieur/intérieur de l'œil selon la topologie MediaPipe.
    160,  # Point situé sur la paupière supérieure.
    158,  # Autre point de la paupière supérieure.
    133,  # Deuxième coin de l'œil.
    153,  # Point situé sur la paupière inférieure.
    144,  # Autre point situé sur la paupière inférieure.
]


# Même principe pour l'œil droit.
RIGHT_EYE = [
    362,  # Coin de l'œil droit.
    385,  # Paupière supérieure.
    387,  # Paupière supérieure.
    263,  # Deuxième coin de l'œil.
    373,  # Paupière inférieure.
    380,  # Paupière inférieure.
]


# Points caractéristiques de la bouche.
MOUTH = {
    "left": 61,      # Coin gauche de la bouche.
    "right": 291,    # Coin droit de la bouche.
    "top": 13,       # Partie supérieure des lèvres.
    "bottom": 14,    # Partie inférieure des lèvres.
}


# Points pouvant être utilisés plus tard pour estimer la pose de la tête.
POSE_IDS = [
    1,    # Point proche du nez.
    152,  # Bas du menton.
    33,   # Œil gauche.
    263,  # Œil droit.
    61,   # Coin gauche de la bouche.
    291,  # Coin droit de la bouche.
]


# ============================================================
# MODÈLE MEDIAPIPE
# ============================================================

# URL officielle du modèle FaceLandmarker de MediaPipe.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/"
    "face_landmarker/float16/1/"
    "face_landmarker.task"
)


# On choisit l'endroit où le modèle sera enregistré sur l'ordinateur.
# Path.home()
# représente le dossier personnel de l'utilisateur.
# Par exemple sous Windows :
# C:\Users\Sanae\
# Le fichier sera alors enregistré par exemple dans :
# C:\Users\Sanae\.focuslens\face_landmarker.task
_MODEL_PATH = (
    Path.home()
    / ".focuslens"
    / "face_landmarker.task"
)

# ============================================================
# FONCTION DE TÉLÉCHARGEMENT DU MODÈLE
# ============================================================

def _ensure_model() -> Path:
    """
    Vérifie que le modèle MediaPipe existe localement.
    S'il n'existe pas :
        → téléchargement automatique.
    S'il existe déjà :
        → aucun téléchargement.
    La fonction retourne ensuite le chemin du modèle.
    """

    # On teste si le fichier du modèle n'existe pas encore.
    if not _MODEL_PATH.exists():

        # On crée le dossier .focuslens s'il n'existe pas déjà.
        # parents=True :
        # crée également les dossiers parents nécessaires.
        # exist_ok=True :
        # ne provoque pas d'erreur si le dossier existe déjà.
        _MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Message affiché à l'utilisateur.
        print(
            "FocusLens : téléchargement du modèle "
            "FaceLandmarker (~4 Mo, une seule fois)…"
        )

        # Télécharge le fichier depuis _MODEL_URL.
        # Premier argument :
        # URL du fichier.
        # Deuxième argument :
        # emplacement où sauvegarder le fichier.
        urllib.request.urlretrieve(
            _MODEL_URL,
            _MODEL_PATH
        )

    # Que le fichier ait déjà existé ou qu'il vienne d'être téléchargé,
    # on retourne son chemin.
    return _MODEL_PATH


# ============================================================
# BACKEND LEGACY
# ============================================================

class _LegacyBackend:
    """
    Backend utilisant l'ancienne API MediaPipe :

        mp.solutions.face_mesh.FaceMesh

    Cette API était très répandue dans les anciennes versions
    de MediaPipe.
    """

    def __init__(
        self,
        det_conf: float,
        trk_conf: float
    ):
        """
        Initialise le modèle FaceMesh.
        det_conf :
            seuil minimum de confiance pour détecter un visage.
        trk_conf :
            seuil minimum de confiance pour suivre le visage entre plusieurs frames.
        """
        
        # Création du modèle FaceMesh.
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            
            # False car on travaille avec une vidéo / webcam.
            # MediaPipe peut donc suivre le visage d'une frame à l'autre.
            static_image_mode=False,
            # On ne souhaite détecter qu'un seul visage.
            max_num_faces=1,
            # Active des landmarks plus précis, notamment autour des yeux, lèvres et iris.
            refine_landmarks=True,
            # Seuil minimum pour accepter une détection.
            min_detection_confidence=det_conf,
            # Seuil minimum pour accepter le tracking du visage.
            min_tracking_confidence=trk_conf
        )

    def process(
        self,
        frame_bgr: np.ndarray ) -> np.ndarray | None:
        """
        Analyse une image BGR.
        Retourne :
            tableau NumPy (N, 3)
        avec :
            x, y, z   ou : None si aucun visage n'est détecté.
        """
        # frame_bgr.shape peut être par exemple :
        # (720, 1280, 3) donc : h(height) = 720, w(width) = 1280 -> donc l'image contient 720×1280=921600 pixels
        # 3 -> Couleurs RGB -> l'image a 3 couleurs par pixel , mais on retient pas cette info dans la ligne suivant :2
        h, w = frame_bgr.shape[:2]
        # OpenCV fournit généralement les images en BGR.
        # MediaPipe attend du RGB.(Red, green, Blue)
        # [:, :, ::-1] inverse donc l'ordre des canaux : BGR → RGB , on dit : prend toutes les 720 lignes, toutes les 1280 colonnes, et les 3 canaux inversés
        rgb_frame = frame_bgr[:, :, ::-1]

        # On envoie l'image RGB à MediaPipe.
        res = self._mesh.process(rgb_frame)

        # Si MediaPipe n'a détecté aucun visage,
        # multi_face_landmarks est vide.
        if not res.multi_face_landmarks:

            # On retourne None pour signaler :
            # "aucun visage détecté".
            return None

        # On récupère les landmarks du premier visage.Comme max_num_faces=1,ce sera normalement le seul visage détecté.
        lm = res.multi_face_landmarks[0].landmark
           #res est un objet contenant les résultats de la détection qui représentent les landmarks des visages détectés(multifacelandmarks) mais prend que le premier visage [0]

        """chaque landmark(point) peut être représenté conceptuellement comme :
                   p = {
                      x: ...
                      y: ...
                      z: ... }
        x, y ,z representent les positions du point dans l'image , si mediapipe detecte que :
        x indique ou de trouve le point au niveau de la largeur de l'image
             Si x = 0.2, cela signifie que :le point est à environ 20 % de la largeur de l'image depuis la gauche.
            
                    0       0.5        1
                │--------│---------│    -> x
                gauche    milieu    droite
          
        y indique ou se trouve le point horizontalement, y=0 -> point le plus haut de l'image , y=1 , le plus bas , 0.5 au milieu de l'image
            si p.x = 0.5
            et p;y = 0.3
            x = 50 % de la largeur
            y = 30 % de la hauteur  
                       largeur
                    ─────────────────────→
                    0         0.5       1
                    │          │         │
                    │          ● ← point │
                    │          │         │
                    │          │         │
                    │          │         │
                    1
                    ↓
                    hauteur
       
        x, y donnent les positiondans l'image 2D , z donne une info suplémenatire qui est la profondeur du point 
        Pourquoi x , y , z sont entre 0 et 1 ? car Mediapipe normalise les coordonnées , au lieu de dire x= 360 px , 
        il dit x=0.3 pour montrer le % de position par rapport à l'image, Cela permet au modèle de fonctionner indépendamment de la résolution de l'image.
        Donc peu importe si l'image fait 900x900 pu 1200x750 0.5 signifie toujours le milieu de l'image
       """
        
        # On veut ici récupérer des coordonnées en pixels. Savoir ou est le point est positionné exactement en pixels
        return np.array(

            [
                [
                    p.x * w,  # Conversion x normalisé → pixels.
                    p.y * h,  # Conversion y normalisé → pixels.
                    p.z * w   # Conversion de la profondeur relative.
                ]

                # On effectue cette opération pour chaque landmark dans le visage
                for p in lm
            ],
            # float32 consomme moins de mémoire que float64 et suffit largement ici.
            dtype=np.float32
        )

    def close(self) -> None:
        """
        Ferme proprement le modèle MediaPipe.
        """
        # Libère les ressources utilisées par FaceMesh.
        self._mesh.close()


# ============================================================
# BACKEND MODERNE : MEDIAPIPE TASKS
# ============================================================

class _TasksBackend:
    """
    Backend utilisant la nouvelle API MediaPipe Tasks :
        mediapipe.tasks.python.vision.FaceLandmarker
    """

    def __init__(
        self,
        det_conf: float,
        trk_conf: float
    ):
        """
        Initialise le nouveau modèle FaceLandmarker.
        """
        # Import de l'API Python de MediaPipe Tasks.
        from mediapipe.tasks import python as mp_python

        # Import des outils de vision de MediaPipe Tasks.
        from mediapipe.tasks.python import vision

        # Configuration du FaceLandmarker.
        options = vision.FaceLandmarkerOptions(

            # BaseOptions indique notamment quel modèle utiliser.
            base_options=mp_python.BaseOptions(

                # _ensure_model() vérifie que le modèle existe.
                # str(...) transforme le Path en chaîne de caractères.
                model_asset_path =
                 str ( _ensure_model() )
            ),

            # VIDEO indique que les images arrivent successivement,
            # comme avec une webcam.
            running_mode=vision.RunningMode.VIDEO,
            # On analyse uniquement un visage.
            num_faces=1,
            # Confiance minimale de détection du visage.
            min_face_detection_confidence=det_conf,
            # Confiance minimale indiquant que le visage est réellement présent.
            min_face_presence_confidence=det_conf,
            # Confiance minimale pour le tracking.
            min_tracking_confidence=trk_conf
        )

        # Création réelle du modèle FaceLandmarker.
        self._lm = (
            vision.FaceLandmarker
            .create_from_options(options)
        )

        # Timestamp initial.
        # L'API VIDEO exige que chaque frame possède un timestamp strictement supérieur au précédent.
        self._ts_ms = 0
    
    def process(
        self,
        frame_bgr: np.ndarray
    ) -> np.ndarray | None:
        """
        Analyse une frame avec l'API MediaPipe Tasks.
        """
        # Récupération de la hauteur et de la largeur.
        h, w = frame_bgr.shape[:2]

        # Conversion BGR → RGB.
        # np.ascontiguousarray garantit que le tableau NumPy
        # est stocké de manière contiguë en mémoire.
        # Certaines bibliothèques bas niveau l'exigent.
        rgb = np.ascontiguousarray(
            frame_bgr[:, :, ::-1]
        )
        # Transformation du tableau NumPy en objet Image MediaPipe.
        mp_img = mp.Image(
            # On précise que l'image est au format RGB standard.
            image_format=mp.ImageFormat.SRGB,
            # Les pixels de l'image sont contenus dans rgb.
            data=rgb
        )

        # On augmente le timestamp de 33 millisecondes.
        # 33 ms correspond environ à :1000 / 33 ≈ 30 FPS.
        # Ce timestamp doit être strictement croissant.
        self._ts_ms += 33
        
        # Analyse de l'image par le modèle.
        # detect_for_video reçoit :
        # 1. l'image
        # 2. le timestamp
        res = self._lm.detect_for_video(
            mp_img,
            self._ts_ms
        )

        # Si aucun visage n'est détecté :
        if not res.face_landmarks:
            # On retourne None.
            return None

        # On récupère les landmarks du premier visage.
        lm = res.face_landmarks[0]
        # On transforme les landmarks normalisés en pixels.
        return np.array(

            [
                [
                    p.x * w,  # Position horizontale en pixels.
                    p.y * h,  # Position verticale en pixels.
                    p.z * w   # Profondeur relative.
                ]

                # Pour chaque landmark détecté.
                for p in lm
            ],
            # Tableau NumPy en float32.
            dtype=np.float32
        )
    def close(self) -> None:
        """
        Ferme proprement le FaceLandmarker moderne.
        """
        # Libère les ressources du modèle.
        self._lm.close()


# ============================================================
# CLASSE PUBLIQUE PRINCIPALE
# ============================================================

class FaceLandmarker:
    """
    Classe utilisée par le reste de l'application.
    Elle choisit automatiquement quel backend MediaPipe utiliser.
    L'utilisateur du module n'a donc pas besoin de savoir quelle version de MediaPipe est installée.
    """

    def __init__(
        self,
        # Valeur par défaut :
        # minimum 0.5 de confiance pour détecter un visage.
        min_detection_conf: float = 0.5,
        # Valeur par défaut :
        # minimum 0.5 de confiance pour suivre le visage.
        min_tracking_conf: float = 0.5
    ):

        # On suppose d'abord que l'ancien backend est disponible.
        self.backend_name = "legacy"

        # On essaye d'utiliser l'ancienne API.
        try:

            # hasattr vérifie si l'objet mp contient un attribut appelé "solutions".
            # Cela revient à demander :"mp.solutions existe-t-il ?"
            if not hasattr(mp, "solutions"):
                # Si l'attribut n'existe pas,
                # on provoque volontairement une erreur.
                raise AttributeError

            # Si mp.solutions existe, on initialise l'ancien backend.
            self._backend = _LegacyBackend(

                # Seuil de détection.
                min_detection_conf,

                # Seuil de tracking.
                min_tracking_conf
            )

        # Si quelque chose échoue dans le bloc try...
        # Attention :
        # Exception englobe déjà AttributeError, donc cette écriture est techniquement redondante.
        except (AttributeError, Exception):

            # On indique que le backend moderne est utilisé.
            self.backend_name = "tasks"

            # On initialise l'API MediaPipe Tasks.
            self._backend = _TasksBackend(

                # Seuil de détection.
                min_detection_conf,

                # Seuil de tracking.
                min_tracking_conf
            )

    def process(
        self,
        frame_bgr: np.ndarray
    ) -> np.ndarray | None:
        """
        Analyse une image.

        Cette méthode ne sait pas directement si le backend est Legacy ou Tasks.Elle délègue simplement le travail
        au backend sélectionné précédemment.
        """

        # Appelle :
        # _LegacyBackend.process(...)
        # ou :
        # _TasksBackend.process(...)
        # selon la version disponible.
        return self._backend.process(
            frame_bgr
        )

    def close(self) -> None:
        """
        Ferme proprement le backend utilisé.
        """
        # Même logique de délégation :
        # LegacyBackend.close()
        # ou :
        # TasksBackend.close()
        self._backend.close()