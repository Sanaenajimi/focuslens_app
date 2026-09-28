"""Analyse temporelle : là où un projet junior devient senior.

Une frame ne veut rien dire. Un CLIGNEMENT est une trajectoire dans le temps ;
une SOMNOLENCE est une statistique sur 30 secondes. Ce module gère :

- Calibration : baseline EAR personnelle sur les premières secondes
- Hystérésis : deux seuils + N frames consécutives → pas d'oscillation d'état
- Machine à états oculaire : OPEN → CLOSED, avec durées
- Événements datés : blink, microsleep, yawn, look_away, absent
- Fenêtres glissantes : clignements/min, PERCLOS
- Score composite lissé (EMA)

v2 — SIGNAL D'ENTRÉE AGNOSTIQUE
Le module ne suppose plus que le signal d'ouverture est un EAR. Il accepte :

  * `ear`       : ratio géométrique, TOUJOURS fourni (calibration, contrôle,
                  repli). Seuils RELATIFS à la baseline personnelle.
  * `openness`  : 1 − probabilité "fermé" du CNN, fourni quand le réseau pilote.
                  Seuils ABSOLUS et calibrés à l'entraînement : le réseau a déjà
                  appris la variabilité morphologique, il n'y a rien à
                  recalibrer par personne.

Quand `openness` est fourni, c'est LUI qui pilote la machine à états, la
vélocité et le score. L'EAR continue d'être suivi en parallèle pour mesurer le
taux de désaccord entre les deux sources — un indicateur d'audit, pas une
décision.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .config import FocusLensConfig


@dataclass
class Event:
    kind: str          # blink | long_blink | microsleep | yawn | look_away | absent
    t_start: float
    t_end: float
    severity: str = "info"   # info | modere | critique
    closing_velocity: float | None = None   # pic de vitesse de fermeture (unités/s, négatif)
    opening_velocity: float | None = None   # pic de vitesse de réouverture (unités/s, positif)
    perclos_before: float | None = None     # PERCLOS mesuré juste avant l'événement (0-1)
    source: str = "ear"                     # qui a tranché : "cnn" | "ear"

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


class EMA:
    """Lissage exponentiel : new = α·x + (1−α)·old. Simple, causal, O(1)."""
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class TemporalAnalyzer:
    def __init__(self, cfg: FocusLensConfig, cnn_threshold: float | None = None):
        self.cfg = cfg
        # Seuil de fermeture sur la probabilité CNN, résolu par l'analyzer
        # depuis le checkpoint (calibré) ou depuis la config.
        self.cnn_threshold = cnn_threshold or cfg.cnn_fallback_threshold

        # calibration (toujours sur l'EAR géométrique)
        self._calib_ears: list[float] = []
        self.baseline_ear: float | None = None

        # état yeux (piloté par le signal actif)
        self._eye_state = "open"
        self._below_count = 0
        self._above_count = 0
        self._closed_since: float | None = None
        self._active_signal = "ear"          # "ear" | "cnn" — signal du dernier update
        self._event_source = "ear"           # source au moment où la fermeture a commencé

        # état bouche / pose / présence
        self._mouth_open_since: float | None = None
        self._away_since: float | None = None
        self._absent_since: float | None = None

        # historique
        self.events: list[Event] = []
        self._blink_times: deque[float] = deque()
        self._perclos_buf: deque[tuple[float, bool]] = deque()

        # vélocité
        self._vel_buf: deque[tuple[float, float]] = deque()
        self._prev_sig: float | None = None
        self._prev_t: float | None = None
        self._closing_velocity: float | None = None

        # lissage
        self.score_ema = EMA(cfg.ema_alpha)
        self.ear_ema = EMA(cfg.ema_alpha)
        self.open_ema = EMA(cfg.cnn_ema_alpha)

        # audit : accord entre les deux sources
        self.frames_compared = 0
        self.frames_disagree = 0
        self.frames_by_source = {"cnn": 0, "ear": 0}

    # ───────────────────────── calibration ─────────────────────────
    def _calibrating(self, t: float) -> bool:
        return t < self.cfg.calib_seconds

    def _finish_calibration(self) -> None:
        if self._calib_ears:
            s = sorted(self._calib_ears)
            # 80e percentile : capture l'EAR "yeux grands ouverts" plutôt que la
            # médiane, tirée vers le bas par les clignements pendant la
            # calibration. Critique pour les porteurs de lunettes (EAR de base
            # ~0.10-0.18 au lieu de ~0.28).
            idx = int(0.80 * len(s))
            raw = s[min(idx, len(s) - 1)]
            self.baseline_ear = max(0.07, min(0.42, raw))
        else:
            self.baseline_ear = 0.25

    # ───────────────────────── seuils ─────────────────────────
    @property
    def thr_ear_closed(self) -> float:
        base = self.baseline_ear or 0.25
        return max(self.cfg.ear_absolute_floor, base * self.cfg.ear_ratio_closed)

    @property
    def thr_ear_open(self) -> float:
        base = self.baseline_ear or 0.25
        return base * self.cfg.ear_ratio_open

    @property
    def thr_open_closed(self) -> float:
        """Ouverture en dessous de laquelle l'œil est déclaré fermé (signal CNN)."""
        return 1.0 - self.cnn_threshold

    @property
    def thr_open_open(self) -> float:
        """Ouverture au-dessus de laquelle l'œil est déclaré rouvert (hystérésis)."""
        return min(0.97, self.thr_open_closed + self.cfg.open_hysteresis_margin)

    @property
    def thr_closed(self) -> float:
        """Seuil de fermeture du signal ACTIF (celui affiché et journalisé)."""
        return self.thr_open_closed if self._active_signal == "cnn" else self.thr_ear_closed

    @property
    def thr_open(self) -> float:
        return self.thr_open_open if self._active_signal == "cnn" else self.thr_ear_open

    # ───────────────────────── vélocité ─────────────────────────
    def _velocity_scale(self) -> float:
        """Normalisation du signal pour rendre la vitesse comparable entre
        sources. L'EAR est divisé par la baseline personnelle ; l'ouverture CNN
        vaut déjà 0-1 et n'a rien à normaliser."""
        if self._active_signal == "cnn":
            return 1.0
        return self.baseline_ear or 0.25

    def _update_velocity(self, t: float, sig: float) -> None:
        if self._prev_sig is not None and self._prev_t is not None and t > self._prev_t:
            raw = (sig - self._prev_sig) / (t - self._prev_t)
            self._vel_buf.append((t, raw / self._velocity_scale()))
        self._prev_sig, self._prev_t = sig, t
        while self._vel_buf and t - self._vel_buf[0][0] > self.cfg.velocity_window_s:
            self._vel_buf.popleft()

    def _peak_velocity(self, take_min: bool) -> float | None:
        if not self._vel_buf:
            return None
        vals = [v for _, v in self._vel_buf]
        return min(vals) if take_min else max(vals)

    # ───────────────────────── PERCLOS / sévérité ─────────────────────────
    def _perclos_before(self, closure_start: float | None) -> float:
        """PERCLOS sur la fenêtre glissante, en excluant les frames de la
        fermeture en cours : c'est la tendance de fond QUI PRÉCÉDAIT
        l'événement, pas polluée par lui."""
        if closure_start is None or not self._perclos_buf:
            return 0.0
        prior = [c for tt, c in self._perclos_buf if tt < closure_start]
        if not prior:
            return 0.0
        return sum(prior) / len(prior)

    def _classify_severity(self, kind: str, duration: float, perclos_before: float,
                           closing_velocity: float | None, opening_velocity: float | None) -> str:
        """Deux signaux indépendants, chacun suffisant pour 'critique' :
        1. PERCLOS avant l'événement — la tendance de fond (le TEMPS)
        2. La vélocité de fermeture/ouverture — la dynamique (la VITESSE)
        Un œil fatigué se ferme ET se rouvre lentement, indépendamment de la
        durée totale : un simple seuil de durée ne capture pas ça."""
        if kind == "blink":
            return "info"
        if perclos_before >= self.cfg.perclos_severity_threshold:
            return "critique"
        slow_close = closing_velocity is not None and abs(closing_velocity) < self.cfg.velocity_slow_threshold
        slow_open = opening_velocity is not None and abs(opening_velocity) < self.cfg.velocity_slow_threshold
        if slow_close and slow_open:
            return "critique"
        return "modere"

    def _live_alert_level(self, t: float) -> str:
        """Niveau d'alerte pendant que l'œil est ENCORE fermé (pas seulement
        une fois l'événement clos, cf. _classify_severity). "none" avant
        microsleep_s : un clignement normal n'est jamais pénalisé."""
        if self._eye_state != "closed" or self._closed_since is None:
            return "none"
        dur = t - self._closed_since
        if dur < self.cfg.microsleep_s:
            return "none"
        if dur >= self.cfg.alert_red_s:
            return "rouge"
        perclos_before = self._perclos_before(self._closed_since)
        slow_closing = (self._closing_velocity is not None and
                        abs(self._closing_velocity) < self.cfg.velocity_slow_threshold)
        if (dur >= self.cfg.alert_orange_s or perclos_before >= self.cfg.perclos_severity_threshold
                or slow_closing):
            return "orange"
        return "jaune"

    # ───────────────────────── update par frame ─────────────────────────
    def update(self, t: float, face_found: bool, ear: float | None,
               mar: float | None, yaw: float | None, pitch: float | None,
               openness: float | None = None, source: str = "ear") -> dict:
        """`ear` est toujours transmis (géométrie). `openness` ne l'est que
        lorsque le CNN pilote : dans ce cas c'est lui qui décide."""
        cfg = self.cfg
        self._active_signal = "cnn" if openness is not None else "ear"

        # -- calibration en cours : on ne mesure que la baseline EAR --
        if self._calibrating(t):
            if ear is not None:
                self._calib_ears.append(ear)
            if openness is not None:
                self.open_ema.update(openness)
            return {"phase": "calibration", "score": None, "eye_state": "open",
                    "perclos": 0.0, "blinks_per_min": 0.0,
                    "thr_closed": None, "microsleep": False, "source": source,
                    "openness": openness, "closed_duration": 0.0,
                    "alert_level": "none", "yawning": False}
        if self.baseline_ear is None:
            self._finish_calibration()

        # -- présence --
        if not face_found:
            if self._absent_since is None:
                self._absent_since = t
        else:
            if self._absent_since is not None:
                self.events.append(Event("absent", self._absent_since, t))
                self._absent_since = None

        microsleep_now = False
        eyes_closed = False
        sig_s: float | None = None

        if face_found and (ear is not None or openness is not None):
            # Les deux canaux sont lissés à chaque frame, même celui qui ne
            # décide pas : c'est ce qui permet de mesurer le désaccord.
            ear_s = self.ear_ema.update(ear) if ear is not None else None
            open_s = self.open_ema.update(openness) if openness is not None else None

            if open_s is not None:
                sig_s, thr_c, thr_o = open_s, self.thr_open_closed, self.thr_open_open
            else:
                sig_s, thr_c, thr_o = ear_s, self.thr_ear_closed, self.thr_ear_open

            self._update_velocity(t, sig_s)
            self.frames_by_source[source] = self.frames_by_source.get(source, 0) + 1

            # -- audit : les deux sources voient-elles la même chose ? --
            if ear_s is not None and open_s is not None:
                self.frames_compared += 1
                if (ear_s < self.thr_ear_closed) != (open_s < self.thr_open_closed):
                    self.frames_disagree += 1

            # -- machine à états yeux avec hystérésis --
            if self._eye_state == "open":
                self._below_count = self._below_count + 1 if sig_s < thr_c else 0
                if self._below_count >= cfg.hysteresis_frames:
                    self._eye_state, self._closed_since = "closed", t
                    self._closing_velocity = self._peak_velocity(take_min=True)
                    self._event_source = source
                    self._below_count = 0
            else:  # closed
                eyes_closed = True
                dur = t - (self._closed_since or t)
                if dur >= cfg.microsleep_s:
                    microsleep_now = True
                self._above_count = self._above_count + 1 if sig_s > thr_o else 0
                if self._above_count >= cfg.hysteresis_frames:
                    kind = "blink" if dur <= cfg.blink_max_s else \
                           ("microsleep" if dur >= cfg.microsleep_s else "long_blink")
                    perclos_before = self._perclos_before(self._closed_since)
                    opening_velocity = self._peak_velocity(take_min=False)
                    severity = self._classify_severity(
                        kind, dur, perclos_before, self._closing_velocity, opening_velocity)
                    self.events.append(Event(kind, self._closed_since, t, severity,
                                             self._closing_velocity, opening_velocity,
                                             perclos_before, self._event_source))
                    if kind == "blink":
                        self._blink_times.append(t)
                    self._eye_state, self._closed_since = "open", None
                    self._above_count = 0
                    self._closing_velocity = None

            # -- bâillement (durée minimale : exclut la parole) --
            if mar is not None and mar > cfg.mar_yawn:
                if self._mouth_open_since is None:
                    self._mouth_open_since = t
            else:
                if self._mouth_open_since is not None:
                    if t - self._mouth_open_since >= cfg.yawn_min_s:
                        self.events.append(Event("yawn", self._mouth_open_since, t))
                    self._mouth_open_since = None

            # -- distraction pose (hystérésis temporelle) --
            away = yaw is not None and (abs(yaw) > cfg.yaw_away_deg or
                                        (pitch is not None and pitch < -cfg.pitch_down_deg))
            if away:
                if self._away_since is None:
                    self._away_since = t
            else:
                if self._away_since is not None:
                    if t - self._away_since >= cfg.pose_away_min_s:
                        self.events.append(Event("look_away", self._away_since, t))
                    self._away_since = None

        # -- fenêtres glissantes --
        while self._blink_times and t - self._blink_times[0] > cfg.rate_window_s:
            self._blink_times.popleft()
        self._perclos_buf.append((t, eyes_closed or not face_found))
        while self._perclos_buf and t - self._perclos_buf[0][0] > cfg.perclos_window_s:
            self._perclos_buf.popleft()
        perclos = sum(c for _, c in self._perclos_buf) / max(1, len(self._perclos_buf))
        elapsed_min = min(cfg.rate_window_s, max(t, 1e-6)) / 60.0
        bpm = len(self._blink_times) / elapsed_min

        # -- score composite --
        score = 0.0
        if face_found:
            score += cfg.w_presence
            if yaw is not None:
                score += cfg.w_facing * max(0.0, 1 - abs(yaw) / (2 * cfg.yaw_away_deg))
            if self._active_signal == "cnn" and self.open_ema.value is not None:
                # L'ouverture CNN est déjà bornée 0-1 : elle s'utilise telle quelle.
                score += cfg.w_eyes * min(1.0, max(0.0, self.open_ema.value))
            elif self.ear_ema.value is not None and self.baseline_ear:
                score += cfg.w_eyes * min(1.0, max(0.0, self.ear_ema.value / self.baseline_ear))
            score += cfg.w_blink_rhythm if cfg.blink_healthy_min <= bpm <= cfg.blink_healthy_max \
                     else cfg.w_blink_rhythm * 0.5
        # -- pénalité de fermeture continue (remplace l'ancien clamp instantané) --
        ramp = 0.0
        if self._eye_state == "closed" and self._closed_since is not None:
            ramp = min(1.0, (t - self._closed_since) / cfg.score_ramp_s)
        if ramp > 0:
            score = score * (1 - ramp) + cfg.score_floor_microsleep * ramp
        score_s = self.score_ema.update(score)

        alert_level = self._live_alert_level(t)

        return {"phase": "analysis", "score": round(score_s, 1),
                "eye_state": self._eye_state, "perclos": round(perclos, 3),
                "blinks_per_min": round(bpm, 1), "thr_closed": round(self.thr_closed, 3),
                "microsleep": microsleep_now, "source": source,
                "openness": round(self.open_ema.value, 3) if self.open_ema.value is not None else None,
                "closed_duration": round(t - self._closed_since, 2) if self._closed_since else 0.0,
                "alert_level": alert_level, "yawning": self._mouth_open_since is not None}

    # ───────────────────────── audit ─────────────────────────
    @property
    def agreement_rate(self) -> float | None:
        """Part des frames où EAR et CNN concluent la même chose. None si une
        seule source a tourné. À citer dans le rapport : un taux qui s'effondre
        signale une caméra, une lumière ou des lunettes hors du domaine
        d'entraînement du réseau."""
        if not self.frames_compared:
            return None
        return 1.0 - self.frames_disagree / self.frames_compared

    def finalize(self, t_end: float) -> None:
        """Clôt les épisodes encore ouverts en fin de vidéo."""
        if self._closed_since is not None:
            dur = t_end - self._closed_since
            kind = "microsleep" if dur >= self.cfg.microsleep_s else "blink"
            self.events.append(Event(kind, self._closed_since, t_end,
                                     source=self._event_source))
        if self._absent_since is not None:
            self.events.append(Event("absent", self._absent_since, t_end))
        if self._away_since is not None and t_end - self._away_since >= self.cfg.pose_away_min_s:
            self.events.append(Event("look_away", self._away_since, t_end))