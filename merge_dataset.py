"""Regroupe des dossiers de classe épars (ex: un dataset déjà pré-découpé en
test/train/val, chacun avec ses sous-dossiers open/closed ou awake/sleepy) en
deux dossiers plats : train/data/open et train/data/closed.

Pourquoi ignorer le découpage fourni par le dataset : la plupart des mirroirs
Kaggle font un split ALÉATOIRE sur les images, pas par sujet. Des frames quasi
identiques du même œil se retrouvent alors à la fois dans leur "train" et leur
"test". En tout regroupant, on laisse train_eye_cnn.py appliquer SA découpe par
sujet, la seule qui protège contre cette fuite de données.

Utilise des liens physiques (os.link) quand c'est possible : aucun espace
disque supplémentaire sur le même volume. Bascule sur une copie sinon (Windows
refuse parfois les liens entre certains montages, ou entre disques différents).

    python merge_dataset.py --root train/data_raw/data ^
        --open train\\data_raw\\data\\train\\awake train\\data_raw\\data\\test\\awake train\\data_raw\\data\\val\\awake ^
        --closed train\\data_raw\\data\\train\\sleepy train\\data_raw\\data\\test\\sleepy train\\data_raw\\data\\val\\sleepy

(La commande exacte à lancer est imprimée par inspect_dataset.py — copie-la
telle quelle plutôt que de la retaper.)
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def link_or_copy(src: Path, dst: Path) -> str:
    """Lien physique si possible (gratuit en espace disque), copie sinon."""
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def merge(sources: list[Path], dst_dir: Path) -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)
    counts = {"link": 0, "copy": 0, "skipped_duplicate": 0}
    seen_names: set[str] = set()

    for src_dir in sources:
        src_dir = Path(src_dir)
        if not src_dir.exists():
            print(f"   ⚠ Dossier introuvable, ignoré : {src_dir}")
            continue
        for p in sorted(src_dir.rglob("*")):
            if p.suffix.lower() not in IMG_EXT:
                continue
            name = p.name
            if name in seen_names:
                # Collision de nom entre test/train/val : préfixe par le
                # dossier parent pour ne perdre aucune image.
                name = f"{p.parent.name}_{p.name}"
            seen_names.add(name)
            dst = dst_dir / name
            if dst.exists():
                counts["skipped_duplicate"] += 1
                continue
            kind = link_or_copy(p, dst)
            counts[kind] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="racine du dataset téléchargé (informatif)")
    ap.add_argument("--open", nargs="+", required=True, help="un ou plusieurs dossiers = classe ouvert/éveillé")
    ap.add_argument("--closed", nargs="+", required=True, help="un ou plusieurs dossiers = classe fermé/endormi")
    ap.add_argument("--dst", default="train/data", help="destination (défaut: train/data)")
    args = ap.parse_args()

    dst = Path(args.dst)
    print(f"Fusion vers {dst}/open et {dst}/closed ...")

    c_open = merge([Path(p) for p in args.open], dst / "open")
    c_closed = merge([Path(p) for p in args.closed], dst / "closed")

    print(f"\nOUVERT  : {sum(c_open.values()):,} fichiers "
          f"({c_open['link']} liens, {c_open['copy']} copies, {c_open['skipped_duplicate']} doublons ignorés)")
    print(f"FERMÉ   : {sum(c_closed.values()):,} fichiers "
          f"({c_closed['link']} liens, {c_closed['copy']} copies, {c_closed['skipped_duplicate']} doublons ignorés)")
    print(f"\n✅ Prêt pour : python train/train_eye_cnn.py --data {dst}")
    print("   Lance d'abord inspect_dataset.py sur ce nouveau dossier pour vérifier")
    print("   le taux de reconnaissance des sujets avant l'entraînement complet.")


if __name__ == "__main__":
    main()
