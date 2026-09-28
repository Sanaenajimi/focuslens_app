"""Configuration FocusLens — chaque seuil est documenté et justifié.

Règle : aucun nombre magique dans le code métier. Tout est ici, modifiable,
traçable, et sérialisable dans le rapport (reproductibilité).

CHANGEMENT MAJEUR (v2) : le CNN entraîné devient la source PRINCIPALE de la
décision œil ouvert/fermé. L'EAR géométrique reste calculé en permanence, mais
son rôle change : calibration personnelle, canal de contrôle (taux de désaccord)
et repli automatique si le modèle est indisponible.
"""
from dataclasses import dataclass, asdict


@dataclass
class FocusLensConfig:
    # --- Vidéo ---
    max_width: int = 960          # redimensionnement borné : précision suffisante, 3-4x plus rapide
    fallback_fps: float = 30.0    # certaines vidéos/webcams ne déclarent pas leur fps

    # --- Détection visage ---
    min_detection_conf: float = 0.5
    min_tracking_conf: float = 0.5

    # ═══════════════ SOURCE DE DÉCISION OŒIL OUVERT / FERMÉ ═══════════════
    # "cnn"  : le réseau entraîné décide (défaut). Repli EAR si le modèle manque.
    # "auto" : le CNN décide, sauf quand il est incertain (proba dans la bande
    #          ambiguë) — là c'est l'EAR géométrique qui tranche.
    # "ear"  : géométrie seule. Aucun modèle chargé, aucune dépendance torch.
    eye_source: str = "cnn"

    # Seuil de décision du CNN sur la probabilité "fermé".
    # None → on lit le seuil CALIBRÉ écrit dans le checkpoint par
    # train_eye_cnn.py (point de la courbe ROC atteignant `target_recall`).
    # Un 0.5 codé en dur est un choix arbitraire : en sécurité routière, rater
    # un œil fermé coûte bien plus cher qu'une alarme de trop.
    cnn_threshold: float | None = None
    cnn_fallback_threshold: float = 0.42   # utilisé si le checkpoint n'en contient pas

    # Lissage EMA de la probabilité CNN, sur ~3-5 frames.
    # Indispensable : un classifieur sature à 0 ou 1, donc son signal brut est
    # un créneau. Lissé, il redevient une courbe dont la PENTE est exploitable
    # — c'est ce qui rend la vitesse de fermeture mesurable côté CNN.
    cnn_ema_alpha: float = 0.35

    # Bande d'incertitude du CNN, utilisée uniquement en mode "auto".
    cnn_uncertain_low: float = 0.35
    cnn_uncertain_high: float = 0.65

    # --- Seuils sur le signal d'ouverture CNN (openness = 1 − p_fermé) ---
    # ABSOLUS, pas relatifs à une baseline : le réseau a déjà appris la
    # variabilité morphologique sur des milliers de sujets, il n'y a rien à
    # recalibrer par personne. L'hystérésis se fait par une marge.
    open_hysteresis_margin: float = 0.14

    # --- Seuils EAR (mode "ear", repli, et canal de contrôle) ---
    # RELATIFS à la baseline personnelle mesurée en calibration :
    # fermé si EAR < ratio_closed * baseline ; réouvert si EAR > ratio_open * baseline
    calib_seconds: float = 5.0
    ear_ratio_closed: float = 0.60        # lunettes : compriment l'EAR de base
    ear_ratio_open: float = 0.75          # hystérésis
    ear_absolute_floor: float = 0.06

    # --- Dynamique palpébrale ---
    blink_max_s: float = 0.55            # clignement normal jusqu'à 0.55 s
    microsleep_s: float = 1.20           # micro-sommeil dès 1.2 s
    hysteresis_frames: int = 2

    # --- MAR (bâillement) ---
    mar_yawn: float = 0.60
    yawn_min_s: float = 0.8            # bouche ouverte < 0.8 s ≠ bâillement (parole)

    # --- Pose de tête ---
    yaw_away_deg: float = 25.0
    pitch_down_deg: float = 20.0
    pose_away_min_s: float = 1.0

    # --- Fenêtres temporelles ---
    ema_alpha: float = 0.25
    perclos_window_s: float = 30.0
    perclos_severity_threshold: float = 0.15

    # --- Vélocité de fermeture/ouverture ---
    # Unité : fractions du signal d'ouverture par seconde. Pour l'EAR le signal
    # est normalisé par la baseline personnelle ; pour le CNN il vaut déjà 0-1.
    # Dans les deux cas une fermeture complète en 0.35 s donne ~3/s : le seuil
    # ci-dessous sépare la paupière qui TOMBE du clignement volontaire.
    velocity_window_s: float = 0.35
    velocity_slow_threshold: float = 3.0
    rate_window_s: float = 60.0

    # --- Score d'attention (pondérations, somme = 100) ---
    w_presence: float = 30.0
    w_facing: float = 30.0
    w_eyes: float = 25.0
    w_blink_rhythm: float = 15.0
    blink_healthy_min: int = 6
    blink_healthy_max: int = 32
    score_floor_microsleep: float = 20.0

    # --- Alerte micro-sommeil temps réel (jaune/orange/rouge) ---
    # score_ramp_s : durée, depuis le début de la fermeture, sur laquelle le
    # score glisse continûment vers le plancher (remplace l'ancien clamp
    # discret déclenché au seuil microsleep_s).
    score_ramp_s: float = 1.2
    alert_orange_s: float = 2.2
    alert_red_s: float = 4.0

    def to_dict(self) -> dict:
        return asdict(self)