"""Exporte train/models/eye_cnn.pt vers le format que la page web peut charger.

    python export_onnx.py

Produit, à côté de la page web (ou dans le dossier que tu indiques) :
  - eye_cnn.onnx       le réseau, dans un format lisible par le navigateur
  - eye_cnn_meta.json  seuil, température, taille d'image, padding, classes —
                        tout ce que la page doit savoir pour prétraiter les
                        images exactement comme à l'entraînement

Lance ce script une seule fois après chaque réentraînement du modèle.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train.models.eye_cnn import EyeCNN, CROP_PAD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="train/models/eye_cnn.pt")
    ap.add_argument("--out-dir", default="webapp", help="dossier où écrire les 2 fichiers")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Introuvable : {ckpt_path}. Lance depuis la racine du projet.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    img_size = int(ckpt.get("img_size", 64))
    width = int(ckpt.get("width", 32))

    model = EyeCNN(img_size=img_size, width=width)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "eye_cnn.onnx"
    meta_path = out_dir / "eye_cnn_meta.json"

    dummy = torch.zeros(2, 1, img_size, img_size)  # batch=2 : un par œil, comme en Python
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # L'exporteur récent de PyTorch externalise parfois les poids dans un
    # fichier .onnx.data séparé. onnxruntime-web, dans le navigateur, ne sait
    # pas aller chercher ce fichier tout seul quand on lui donne juste l'URL
    # du .onnx (contrairement à Python, qui le fait automatiquement sur
    # disque) — ça provoque une erreur interne illisible ("X is not a
    # function") au chargement côté page. On refusionne donc tout dans un
    # seul fichier, ce qui est de toute façon inutile à séparer pour un
    # modèle de cette taille (quelques dizaines de milliers de paramètres).
    import onnx
    model_proto = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save(model_proto, str(onnx_path), save_as_external_data=False)
    stray_data = onnx_path.with_suffix(".onnx.data")
    if stray_data.exists():
        stray_data.unlink()

    meta = {
        "img_size": img_size,
        "width": width,
        "crop_pad": float(ckpt.get("crop_pad", CROP_PAD)),
        "threshold": ckpt.get("threshold"),
        "temperature": float(ckpt.get("temperature", 1.0)),
        "class_to_idx": ckpt.get("class_to_idx", {"closed": 0, "open": 1}),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"✅ Exporté :")
    print(f"   {onnx_path}  ({onnx_path.stat().st_size / 1024:.0f} Ko)")
    print(f"   {meta_path}")
    print(f"   seuil={meta['threshold']}, température={meta['temperature']:.3f}, "
          f"img_size={img_size}, classes={meta['class_to_idx']}")
    print(f"\n   Place ces deux fichiers dans le même dossier que index.html, puis")
    print(f"   sers la page via : python -m http.server 8000")


if __name__ == "__main__":
    main()