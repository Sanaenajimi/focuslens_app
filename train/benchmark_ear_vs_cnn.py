"""Comparaison EAR vs CNN sur la MÊME vidéo — l'outil à montrer au jury.

    python train/benchmark_ear_vs_cnn.py --video demo.mp4
    python train/benchmark_ear_vs_cnn.py --video demo.mp4 --truth demo_truth.csv

Le principe : rejouer exactement la même séquence avec chaque source de
décision, puis comparer trois choses que le jury peut vérifier à l'œil.

  1. LE SIGNAL       — les deux courbes superposées dans le temps. On voit
                       immédiatement où elles divergent et pourquoi.
  2. LES ÉVÉNEMENTS  — combien de micro-sommeils, à quel instant, avec quelle
                       sévérité. C'est la sortie qui compte vraiment ; deux
                       sources peuvent différer de 4% sur les frames et
                       produire le même verdict, ou l'inverse.
  3. LE COÛT         — le temps de traitement par image. Un gain de rappel qui
                       divise le fps par trois n'est pas un gain.

Avec `--truth`, un CSV `t_start,t_end` des fermetures réellement observées, le
script calcule en plus le rappel et les fausses alarmes de chaque source AU
NIVEAU ÉVÉNEMENT — la seule métrique qui ait un sens pour de la somnolence.
Sans vérité terrain, il rend l'accord entre sources et les écarts de timing.

Sorties dans `bench/` : signals.png, events.csv, summary.json, report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analyzer import FocusLens
from src.config import FocusLensConfig


# ═════════════════════════ passage vidéo ═════════════════════════

def run_pass(video: str, source: str, max_seconds: float | None) -> dict:
    """Rejoue la vidéo avec une source de décision donnée."""
    cfg = FocusLensConfig()
    cfg.eye_source = source
    fl = FocusLens(cfg)
    if source in ("cnn", "auto") and cfg.eye_source == "ear":
        print(f"   ⚠  {fl.cnn_status}")

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 240:
        fps = cfg.fallback_fps

    rows, latencies = [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        if max_seconds and t >= max_seconds:
            break
        t0 = time.perf_counter()
        res = fl.process(frame, t)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        rows.append(res)
        i += 1
    duration = i / fps
    cap.release()
    fl.finalize(duration)

    return {
        "source": fl.cfg.eye_source,          # peut avoir basculé sur "ear" en repli
        "requested": source,
        "status": fl.cnn_status,
        "rows": rows,
        "events": fl.temporal.events,
        "duration": duration,
        "fps_video": fps,
        "ms_per_frame": float(np.median(latencies)) if latencies else 0.0,
        "ms_p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "summary": fl.source_summary(),
    }


# ═════════════════════════ comparaisons ═════════════════════════

def closed_mask(rows, dt: float, duration: float) -> np.ndarray:
    """Vecteur booléen 'yeux fermés' échantillonné à pas fixe."""
    n = max(1, int(duration / dt))
    grid = np.zeros(n, dtype=bool)
    for r in rows:
        k = int(r.t / dt)
        if 0 <= k < n:
            grid[k] = grid[k] or (r.eye_state == "closed")
    return grid


def intervals_from_events(events, kinds=("microsleep", "long_blink")):
    return [(e.t_start, e.t_end, e.kind, e.severity) for e in events if e.kind in kinds]


def match_events(pred, truth, tolerance: float = 0.75):
    """Un événement prédit compte comme détecté s'il chevauche un intervalle
    vrai, à `tolerance` seconde près sur les bords."""
    matched, used = 0, set()
    for ps, pe, *_ in pred:
        for j, (ts, te) in enumerate(truth):
            if j in used:
                continue
            if ps <= te + tolerance and pe >= ts - tolerance:
                matched += 1
                used.add(j)
                break
    return matched, len(used)


def load_truth(path: str) -> list[tuple[float, float]]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append((float(row["t_start"]), float(row["t_end"])))
    return sorted(out)


# ═════════════════════════ figure ═════════════════════════

def plot_signals(ear_run, cnn_run, truth, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1]})

    # 1) EAR géométrique, avec son seuil personnel
    t_e = [r.t for r in ear_run["rows"] if r.ear is not None]
    v_e = [r.ear for r in ear_run["rows"] if r.ear is not None]
    axes[0].plot(t_e, v_e, lw=1.1, color="#12988C")
    thr_e = [r.thr_closed for r in ear_run["rows"] if r.thr_closed]
    if thr_e:
        axes[0].axhline(np.median(thr_e), ls="--", lw=1, color="#DC4B39",
                        label=f"seuil EAR {np.median(thr_e):.3f}")
    axes[0].set_ylabel("EAR")
    axes[0].set_title("Signal géométrique — calibré sur la morphologie du conducteur")
    axes[0].legend(loc="upper right", fontsize=8)

    # 2) ouverture CNN lissée, avec son seuil calibré
    t_c = [r.t for r in cnn_run["rows"] if r.openness is not None]
    v_c = [r.openness for r in cnn_run["rows"] if r.openness is not None]
    if t_c:
        axes[1].plot(t_c, v_c, lw=1.1, color="#1E3A8A")
        thr_c = [r.thr_closed for r in cnn_run["rows"] if r.thr_closed]
        if thr_c:
            axes[1].axhline(np.median(thr_c), ls="--", lw=1, color="#DC4B39",
                            label=f"seuil CNN {np.median(thr_c):.3f}")
        axes[1].legend(loc="upper right", fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "Mode CNN indisponible sur cette machine",
                     ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_ylabel("Ouverture CNN")
    axes[1].set_title("Signal appris — probabilité lissée, seuil calibré à l'entraînement")

    # 3) bandes d'événements : vérité, EAR, CNN
    lanes = [("Vérité", truth or [], "#333"),
             ("EAR", [(a, b) for a, b, *_ in intervals_from_events(ear_run["events"])], "#12988C"),
             ("CNN", [(a, b) for a, b, *_ in intervals_from_events(cnn_run["events"])], "#1E3A8A")]
    for k, (name, spans, col) in enumerate(lanes):
        for a, b in spans:
            axes[2].barh(k, max(b - a, 0.08), left=a, height=0.55, color=col)
    axes[2].set_yticks(range(len(lanes)))
    axes[2].set_yticklabels([n for n, _, _ in lanes])
    axes[2].set_xlabel("temps (s)")
    axes[2].set_title("Fermetures prolongées détectées")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ═════════════════════════ main ═════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--truth", help="CSV avec colonnes t_start,t_end des fermetures réelles")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--out", default="bench")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    truth = load_truth(args.truth) if args.truth else []

    print("▸ Passage 1/2 : EAR géométrique")
    ear_run = run_pass(args.video, "ear", args.max_seconds)
    print("▸ Passage 2/2 : CNN entraîné")
    cnn_run = run_pass(args.video, "cnn", args.max_seconds)

    duration = min(ear_run["duration"], cnn_run["duration"])
    dt = 1.0 / 10.0
    m_ear = closed_mask(ear_run["rows"], dt, duration)
    m_cnn = closed_mask(cnn_run["rows"], dt, duration)
    n = min(len(m_ear), len(m_cnn))
    agreement = float((m_ear[:n] == m_cnn[:n]).mean()) if n else 0.0

    ev_ear = intervals_from_events(ear_run["events"])
    ev_cnn = intervals_from_events(cnn_run["events"])

    summary = {
        "video": args.video,
        "duration_s": round(duration, 2),
        "agreement_frame_level": round(agreement, 4),
        "ear": {
            "events": len(ev_ear),
            "critiques": sum(1 for *_, sev in ev_ear if sev == "critique"),
            "ms_per_frame": round(ear_run["ms_per_frame"], 2),
            "ms_p95": round(ear_run["ms_p95"], 2),
        },
        "cnn": {
            "events": len(ev_cnn),
            "critiques": sum(1 for *_, sev in ev_cnn if sev == "critique"),
            "ms_per_frame": round(cnn_run["ms_per_frame"], 2),
            "ms_p95": round(cnn_run["ms_p95"], 2),
            "status": cnn_run["status"],
            "share_cnn_frames": cnn_run["summary"]["share_cnn"],
            "agreement_internal": cnn_run["summary"]["agreement_rate"],
        },
    }

    if truth:
        for key, ev in (("ear", ev_ear), ("cnn", ev_cnn)):
            hit, covered = match_events(ev, truth)
            summary[key]["recall_events"] = round(covered / max(1, len(truth)), 3)
            summary[key]["false_alarms"] = max(0, len(ev) - hit)
        summary["truth_events"] = len(truth)

    # ── sorties ──
    plot_signals(ear_run, cnn_run, truth, out / "signals.png")

    with open(out / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "kind", "t_start", "t_end", "duree_s", "severite",
                    "vitesse_fermeture", "vitesse_ouverture", "perclos_avant"])
        for name, run in (("ear", ear_run), ("cnn", cnn_run)):
            for e in run["events"]:
                w.writerow([name, e.kind, round(e.t_start, 2), round(e.t_end, 2),
                            round(e.duration, 2), e.severity,
                            None if e.closing_velocity is None else round(e.closing_velocity, 2),
                            None if e.opening_velocity is None else round(e.opening_velocity, 2),
                            None if e.perclos_before is None else round(e.perclos_before, 3)])

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    (out / "report.md").write_text(build_report(summary), encoding="utf-8")

    print(f"\n✅ Comparaison écrite dans {out}/")
    print(f"   Accord entre sources : {agreement*100:.1f}% des frames")
    print(f"   Événements  EAR {len(ev_ear)}  ·  CNN {len(ev_cnn)}")
    print(f"   Coût/frame  EAR {summary['ear']['ms_per_frame']} ms  ·  "
          f"CNN {summary['cnn']['ms_per_frame']} ms")


def build_report(s: dict) -> str:
    e, c = s["ear"], s["cnn"]
    lines = [
        "# Comparaison EAR vs CNN",
        "",
        f"Vidéo : `{s['video']}` · {s['duration_s']} s",
        "",
        "| | EAR géométrique | CNN entraîné |",
        "|---|---|---|",
        f"| Fermetures prolongées détectées | {e['events']} | {c['events']} |",
        f"| Dont classées critiques | {e['critiques']} | {c['critiques']} |",
        f"| Temps par image (médiane) | {e['ms_per_frame']} ms | {c['ms_per_frame']} ms |",
        f"| Temps par image (p95) | {e['ms_p95']} ms | {c['ms_p95']} ms |",
    ]
    if "recall_events" in e:
        lines += [
            f"| Rappel sur événements réels | {e['recall_events']*100:.0f} % | {c['recall_events']*100:.0f} % |",
            f"| Fausses alarmes | {e['false_alarms']} | {c['false_alarms']} |",
        ]
    lines += [
        "",
        f"Accord entre les deux sources : **{s['agreement_frame_level']*100:.1f} %** des frames.",
        "",
        "## Lecture",
        "",
        "Un accord élevé avec des comptages d'événements différents signifie que",
        "les sources divergent sur les bords des fermetures, pas sur leur",
        "existence. C'est le cas le plus fréquent, et c'est ce qui décide de la",
        "sévérité : quelques dixièmes de seconde de plus font passer un",
        "clignement long en micro-sommeil.",
        "",
        f"État du modèle : {c['status']}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
