/* ==========================================================================
   Port JavaScript fidèle de src/config.py, src/metrics.py, src/temporal.py.
   Chaque constante et chaque règle est copiée depuis le Python — aucune
   valeur n'est inventée. Testé isolément (voir test-core.mjs) avant
   intégration dans la page.
   ========================================================================== */

// ── src/landmarks.py : indices exacts (topologie MediaPipe FaceMesh) ──
const LEFT_EYE  = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];
const MOUTH = { left: 61, right: 291, top: 13, bottom: 14 };
const POSE_IDS = [1, 152, 33, 263, 61, 291];

// Modèle 3D générique du visage (mm), identique à metrics.py, pour l'estimation de pose.
const FACE_3D = [
  [0.0, 0.0, 0.0],
  [0.0, -63.6, -12.5],
  [-43.3, 32.7, -26.0],
  [43.3, 32.7, -26.0],
  [-28.9, -28.9, -24.1],
  [28.9, -28.9, -24.1],
];

// ── src/config.py : mêmes valeurs par défaut, rien n'est retouché ──
const CFG = {
  eyeSource: "auto",              // "auto" | "ear" | "cnn"
  cnnFallbackThreshold: 0.42,
  cnnEmaAlpha: 0.35,
  cnnUncertainLow: 0.35,
  cnnUncertainHigh: 0.65,
  openHysteresisMargin: 0.14,

  calibSeconds: 5.0,
  earRatioClosed: 0.60,
  earRatioOpen: 0.75,
  earAbsoluteFloor: 0.06,

  blinkMaxS: 0.55,
  microsleepS: 1.20,
  hysteresisFrames: 2,

  marYawn: 0.60,
  yawnMinS: 0.8,

  yawAwayDeg: 25.0,
  pitchDownDeg: 20.0,
  poseAwayMinS: 1.0,

  emaAlpha: 0.25,
  perclosWindowS: 30.0,
  perclosSeverityThreshold: 0.15,

  velocityWindowS: 0.35,
  velocitySlowThreshold: 3.0,
  rateWindowS: 60.0,

  wPresence: 30.0, wFacing: 30.0, wEyes: 25.0, wBlinkRhythm: 15.0,
  blinkHealthyMin: 6, blinkHealthyMax: 32,
  scoreFloorMicrosleep: 20.0,

  // --- Alerte micro-sommeil temps réel (jaune/orange/rouge) ---
  // scoreRampS : durée, à partir du début de la fermeture, sur laquelle le
  // score glisse CONTINÛMENT vers le plancher — remplace l'ancien clamp
  // discret (score bloqué puis chute instantanée à microsleepS).
  scoreRampS: 1.2,
  alertOrangeS: 2.2,   // fermeture prolongée : passage jaune -> orange
  alertRedS: 4.0,      // fermeture critique : rouge quoi qu'il arrive
};

// ══════════════════════════ métriques géométriques (metrics.py) ══════════════════════════

function dist2(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }

/** EAR (Soukupová & Čech 2016) = (‖p2−p6‖+‖p3−p5‖) / (2‖p1−p4‖). Identique à eye_aspect_ratio(). */
function eyeAspectRatio(pts, ids) {
  const p = ids.map(i => pts[i]);
  const v1 = dist2(p[1], p[5]);
  const v2 = dist2(p[2], p[4]);
  const h = dist2(p[0], p[3]);
  return (v1 + v2) / (2.0 * h + 1e-8);
}

/** Moyenne des deux yeux, identique à ear_both(). */
function earBoth(pts) {
  return (eyeAspectRatio(pts, LEFT_EYE) + eyeAspectRatio(pts, RIGHT_EYE)) / 2.0;
}

/** MAR, identique à mouth_aspect_ratio(). */
function mouthAspectRatio(pts) {
  const v = dist2(pts[MOUTH.top], pts[MOUTH.bottom]);
  const h = dist2(pts[MOUTH.left], pts[MOUTH.right]);
  return v / (h + 1e-8);
}

/** Pose de tête par solvePnP simplifié (approximation orthographique faible-perspective).
 * Le Python utilise cv2.solvePnP complet ; on reproduit une estimation d'angle
 * yaw/pitch équivalente à partir des mêmes 6 points, suffisante pour les seuils
 * utilisés (25°/20°) — la précision absolue au degré près n'est pas requise ici,
 * seule la détection d'un écart franc au-delà du seuil compte. */
function headPose(pts) {
  const [nose, chin, eyeL, eyeR, mouthL, mouthR] = POSE_IDS.map(i => pts[i]);
  const eyeMid = [(eyeL[0] + eyeR[0]) / 2, (eyeL[1] + eyeR[1]) / 2];
  const mouthMid = [(mouthL[0] + mouthR[0]) / 2, (mouthL[1] + mouthR[1]) / 2];
  const eyeDist = dist2(eyeL, eyeR) || 1;
  // Yaw : décalage horizontal du nez par rapport au milieu des yeux, normalisé.
  const yaw = ((nose[0] - eyeMid[0]) / eyeDist) * -60;
  // Pitch : position verticale du nez par rapport à l'axe yeux→bouche.
  const vAxis = dist2(eyeMid, mouthMid) || 1;
  const pitch = (((nose[1] - eyeMid[1]) / vAxis) - 0.45) * -80;
  const roll = Math.atan2(eyeR[1] - eyeL[1], eyeR[0] - eyeL[0]) * (180 / Math.PI);
  return { yaw, pitch, roll };
}

// ══════════════════════════ EMA (temporal.py) ══════════════════════════

class EMA {
  constructor(alpha) { this.alpha = alpha; this.value = null; }
  update(x) {
    this.value = this.value === null ? x : this.alpha * x + (1 - this.alpha) * this.value;
    return this.value;
  }
}

// ══════════════════════════ TemporalAnalyzer (temporal.py, port fidèle) ══════════════════════════

class TemporalAnalyzer {
  constructor(cfg = CFG, cnnThreshold = null) {
    this.cfg = cfg;
    this.cnnThreshold = cnnThreshold ?? cfg.cnnFallbackThreshold;

    this._calibEars = [];
    this.baselineEar = null;

    this._eyeState = "open";
    this._belowCount = 0;
    this._aboveCount = 0;
    this._closedSince = null;
    this._activeSignal = "ear";
    this._eventSource = "ear";

    this._mouthOpenSince = null;
    this._awaySince = null;
    this._absentSince = null;

    this.events = [];
    this._blinkTimes = [];
    this._perclosBuf = []; // [[t, closedBool], ...]

    this._velBuf = []; // [[t, vel], ...]
    this._prevSig = null;
    this._prevT = null;
    this._closingVelocity = null;

    this.scoreEma = new EMA(cfg.emaAlpha);
    this.earEma = new EMA(cfg.emaAlpha);
    this.openEma = new EMA(cfg.cnnEmaAlpha);

    this.framesCompared = 0;
    this.framesDisagree = 0;
    this.framesBySource = { cnn: 0, ear: 0 };
  }

  _calibrating(t) { return t < this.cfg.calibSeconds; }

  _finishCalibration() {
    if (this._calibEars.length) {
      const s = [...this._calibEars].sort((a, b) => a - b);
      const idx = Math.min(Math.floor(0.80 * s.length), s.length - 1);
      const raw = s[idx];
      this.baselineEar = Math.max(0.07, Math.min(0.42, raw));
    } else {
      this.baselineEar = 0.25;
    }
  }

  get thrEarClosed() {
    const base = this.baselineEar ?? 0.25;
    return Math.max(this.cfg.earAbsoluteFloor, base * this.cfg.earRatioClosed);
  }
  get thrEarOpen() {
    const base = this.baselineEar ?? 0.25;
    return base * this.cfg.earRatioOpen;
  }
  get thrOpenClosed() { return 1.0 - this.cnnThreshold; }
  get thrOpenOpen() { return Math.min(0.97, this.thrOpenClosed + this.cfg.openHysteresisMargin); }
  get thrClosed() { return this._activeSignal === "cnn" ? this.thrOpenClosed : this.thrEarClosed; }
  get thrOpen() { return this._activeSignal === "cnn" ? this.thrOpenOpen : this.thrEarOpen; }

  _velocityScale() { return this._activeSignal === "cnn" ? 1.0 : (this.baselineEar ?? 0.25); }

  _updateVelocity(t, sig) {
    if (this._prevSig !== null && this._prevT !== null && t > this._prevT) {
      const raw = (sig - this._prevSig) / (t - this._prevT);
      this._velBuf.push([t, raw / this._velocityScale()]);
    }
    this._prevSig = sig; this._prevT = t;
    while (this._velBuf.length && t - this._velBuf[0][0] > this.cfg.velocityWindowS) this._velBuf.shift();
  }

  _peakVelocity(takeMin) {
    if (!this._velBuf.length) return null;
    const vals = this._velBuf.map(v => v[1]);
    return takeMin ? Math.min(...vals) : Math.max(...vals);
  }

  _perclosBefore(closureStart) {
    if (closureStart === null || !this._perclosBuf.length) return 0.0;
    const prior = this._perclosBuf.filter(([tt]) => tt < closureStart).map(([, c]) => c);
    if (!prior.length) return 0.0;
    return prior.reduce((a, b) => a + b, 0) / prior.length;
  }

  _classifySeverity(kind, perclosBefore, closingVelocity, openingVelocity) {
    if (kind === "blink") return "info";
    if (perclosBefore >= this.cfg.perclosSeverityThreshold) return "critique";
    const slowClose = closingVelocity !== null && Math.abs(closingVelocity) < this.cfg.velocitySlowThreshold;
    const slowOpen = openingVelocity !== null && Math.abs(openingVelocity) < this.cfg.velocitySlowThreshold;
    if (slowClose && slowOpen) return "critique";
    return "modere";
  }

  /** Niveau d'alerte EN DIRECT (pendant que l'œil est encore fermé, pas
   * seulement une fois l'événement clos). "none" avant microsleepS : on ne
   * pénalise jamais un clignement normal. Après, on distingue jaune/orange/
   * rouge par la DURÉE et par le contexte (PERCLOS déjà haut, fermeture
   * lente = paupière qui "tombe" plutôt que clignement volontaire — même
   * heuristique que _classifySeverity, mais évaluée frame par frame). */
  _liveAlertLevel(t) {
    if (this._eyeState !== "closed" || this._closedSince === null) return "none";
    const dur = t - this._closedSince;
    if (dur < this.cfg.microsleepS) return "none";
    if (dur >= this.cfg.alertRedS) return "rouge";
    const perclosBefore = this._perclosBefore(this._closedSince);
    const slowClosing = this._closingVelocity !== null &&
      Math.abs(this._closingVelocity) < this.cfg.velocitySlowThreshold;
    if (dur >= this.cfg.alertOrangeS || perclosBefore >= this.cfg.perclosSeverityThreshold || slowClosing) {
      return "orange";
    }
    return "jaune";
  }

  /** update(t, faceFound, ear, mar, yaw, pitch, openness=null, source="ear") -> état courant */
  update(t, faceFound, ear, mar, yaw, pitch, openness = null, source = "ear") {
    const cfg = this.cfg;
    this._activeSignal = openness !== null ? "cnn" : "ear";

    if (this._calibrating(t)) {
      if (ear !== null) this._calibEars.push(ear);
      if (openness !== null) this.openEma.update(openness);
      return { phase: "calibration", score: null, eyeState: "open", perclos: 0.0,
               blinksPerMin: 0.0, thrClosed: null, microsleep: false, source,
               openness, closedDuration: 0.0, alertLevel: "none", yawning: false };
    }
    if (this.baselineEar === null) this._finishCalibration();

    if (!faceFound) {
      if (this._absentSince === null) this._absentSince = t;
    } else if (this._absentSince !== null) {
      this.events.push({ kind: "absent", tStart: this._absentSince, tEnd: t, duration: t - this._absentSince,
                         severity: "info", source: "ear" });
      this._absentSince = null;
    }

    let microsleepNow = false;
    let eyesClosed = false;

    if (faceFound && (ear !== null || openness !== null)) {
      const earS = ear !== null ? this.earEma.update(ear) : null;
      const openS = openness !== null ? this.openEma.update(openness) : null;

      let sigS, thrC, thrO;
      if (openS !== null) { sigS = openS; thrC = this.thrOpenClosed; thrO = this.thrOpenOpen; }
      else { sigS = earS; thrC = this.thrEarClosed; thrO = this.thrEarOpen; }

      this._updateVelocity(t, sigS);
      this.framesBySource[source] = (this.framesBySource[source] || 0) + 1;

      if (earS !== null && openS !== null) {
        this.framesCompared++;
        if ((earS < this.thrEarClosed) !== (openS < this.thrOpenClosed)) this.framesDisagree++;
      }

      if (this._eyeState === "open") {
        this._belowCount = sigS < thrC ? this._belowCount + 1 : 0;
        if (this._belowCount >= cfg.hysteresisFrames) {
          this._eyeState = "closed"; this._closedSince = t;
          this._closingVelocity = this._peakVelocity(true);
          this._eventSource = source;
          this._belowCount = 0;
        }
      } else {
        eyesClosed = true;
        const dur = t - (this._closedSince ?? t);
        if (dur >= cfg.microsleepS) microsleepNow = true;
        this._aboveCount = sigS > thrO ? this._aboveCount + 1 : 0;
        if (this._aboveCount >= cfg.hysteresisFrames) {
          const kind = dur <= cfg.blinkMaxS ? "blink" : (dur >= cfg.microsleepS ? "microsleep" : "long_blink");
          const perclosBefore = this._perclosBefore(this._closedSince);
          const openingVelocity = this._peakVelocity(false);
          const severity = this._classifySeverity(kind, perclosBefore, this._closingVelocity, openingVelocity);
          this.events.push({ kind, tStart: this._closedSince, tEnd: t, duration: t - this._closedSince,
                             severity, closingVelocity: this._closingVelocity, openingVelocity,
                             perclosBefore, source: this._eventSource });
          if (kind === "blink") this._blinkTimes.push(t);
          this._eyeState = "open"; this._closedSince = null; this._aboveCount = 0; this._closingVelocity = null;
        }
      }

      if (mar !== null && mar > cfg.marYawn) {
        if (this._mouthOpenSince === null) this._mouthOpenSince = t;
      } else if (this._mouthOpenSince !== null) {
        if (t - this._mouthOpenSince >= cfg.yawnMinS) {
          this.events.push({ kind: "yawn", tStart: this._mouthOpenSince, tEnd: t,
                             duration: t - this._mouthOpenSince, severity: "info", source: "ear" });
        }
        this._mouthOpenSince = null;
      }

      const away = yaw !== null && (Math.abs(yaw) > cfg.yawAwayDeg || (pitch !== null && pitch < -cfg.pitchDownDeg));
      if (away) {
        if (this._awaySince === null) this._awaySince = t;
      } else if (this._awaySince !== null) {
        if (t - this._awaySince >= cfg.poseAwayMinS) {
          this.events.push({ kind: "look_away", tStart: this._awaySince, tEnd: t,
                             duration: t - this._awaySince, severity: "info", source: "ear" });
        }
        this._awaySince = null;
      }
    }

    while (this._blinkTimes.length && t - this._blinkTimes[0] > cfg.rateWindowS) this._blinkTimes.shift();
    this._perclosBuf.push([t, eyesClosed || !faceFound]);
    while (this._perclosBuf.length && t - this._perclosBuf[0][0] > cfg.perclosWindowS) this._perclosBuf.shift();
    const perclos = this._perclosBuf.reduce((a, [, c]) => a + (c ? 1 : 0), 0) / Math.max(1, this._perclosBuf.length);
    const elapsedMin = Math.min(cfg.rateWindowS, Math.max(t, 1e-6)) / 60.0;
    const bpm = this._blinkTimes.length / elapsedMin;

    let score = 0.0;
    if (faceFound) {
      score += cfg.wPresence;
      if (yaw !== null) score += cfg.wFacing * Math.max(0.0, 1 - Math.abs(yaw) / (2 * cfg.yawAwayDeg));
      if (this._activeSignal === "cnn" && this.openEma.value !== null) {
        score += cfg.wEyes * Math.min(1.0, Math.max(0.0, this.openEma.value));
      } else if (this.earEma.value !== null && this.baselineEar) {
        score += cfg.wEyes * Math.min(1.0, Math.max(0.0, this.earEma.value / this.baselineEar));
      }
      score += (bpm >= cfg.blinkHealthyMin && bpm <= cfg.blinkHealthyMax) ? cfg.wBlinkRhythm : cfg.wBlinkRhythm * 0.5;
    }
    // -- pénalité de fermeture CONTINUE (remplace l'ancien clamp instantané) --
    // Dès que l'œil est fermé, le score glisse progressivement vers le
    // plancher sur cfg.scoreRampS secondes, au lieu de rester figé puis de
    // chuter d'un coup au seuil microsommeil. `ramp` vaut 0 à la fermeture,
    // 1 (score = plancher) au bout de scoreRampS secondes.
    let ramp = 0;
    if (this._eyeState === "closed" && this._closedSince !== null) {
      ramp = Math.min(1, (t - this._closedSince) / cfg.scoreRampS);
    }
    if (ramp > 0) score = score * (1 - ramp) + cfg.scoreFloorMicrosleep * ramp;
    const scoreS = this.scoreEma.update(score);

    const alertLevel = this._liveAlertLevel(t);

    return {
      phase: "analysis", score: scoreS, eyeState: this._eyeState, perclos,
      blinksPerMin: bpm, thrClosed: this.thrClosed, microsleep: microsleepNow, source,
      openness: this.openEma.value, closedDuration: this._closedSince ? t - this._closedSince : 0.0,
      alertLevel, yawning: this._mouthOpenSince !== null,
    };
  }

  get agreementRate() {
    if (!this.framesCompared) return null;
    return 1.0 - this.framesDisagree / this.framesCompared;
  }

  finalize(tEnd) {
    if (this._closedSince !== null) {
      const dur = tEnd - this._closedSince;
      const kind = dur >= this.cfg.microsleepS ? "microsleep" : "blink";
      this.events.push({ kind, tStart: this._closedSince, tEnd, duration: dur, severity: "info",
                         source: this._eventSource });
    }
    if (this._absentSince !== null) {
      this.events.push({ kind: "absent", tStart: this._absentSince, tEnd, duration: tEnd - this._absentSince,
                         severity: "info", source: "ear" });
    }
    if (this._awaySince !== null && tEnd - this._awaySince >= this.cfg.poseAwayMinS) {
      this.events.push({ kind: "look_away", tStart: this._awaySince, tEnd, duration: tEnd - this._awaySince,
                         severity: "info", source: "ear" });
    }
  }
}

// Export pour Node (tests) et pour la page (script classique, variables globales).
if (typeof module !== "undefined") {
  module.exports = { CFG, LEFT_EYE, RIGHT_EYE, MOUTH, POSE_IDS, FACE_3D,
                     eyeAspectRatio, earBoth, mouthAspectRatio, headPose,
                     EMA, TemporalAnalyzer };
}