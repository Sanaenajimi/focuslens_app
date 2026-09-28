"""Diagnostic de l'archive MRL téléchargée : où sont les images, comment sont-elles
réparties, et le nommage porte-t-il un identifiant de sujet exploitable.

Ne déplace rien tout seul — il affiche ce qu'il trouve pour que tu valides avant
de réorganiser. Lance-le une fois l'archive dézippée :

    python3 inspect_dataset.py train/data_raw
"""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
SUBJECT_RE = re.compile(r"^([A-Za-z]?\d+)_")

# mots-clés qui trahissent un dossier de classe, quel que soit le nommage exact.
# "awake"/"sleepy" couvre les mirroirs repackagés pour la détection de somnolence
# (le mapping vers ouvert/fermé est probable mais À CONFIRMER via le readme du
# dataset avant de lancer un entraînement complet dessus).
OPEN_HINTS = {"open", "opened", "awake", "1"}
CLOSED_HINTS = {"closed", "close", "sleepy", "drowsy", "0"}


def main(root: str):
    root = Path(root)
    if not root.exists():
        print(f"✗ Dossier introuvable : {root}")
        return

    all_imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
    print(f"Images trouvées : {len(all_imgs)}")
    if not all_imgs:
        print("Aucune image sous cette racine — vérifie le chemin ou dézippe d'abord.")
        return

    # 1) Quels dossiers contiennent réellement des images, et combien
    by_parent = Counter(p.parent for p in all_imgs)
    print(f"\nDossiers contenant des images ({len(by_parent)}) :")
    for folder, n in sorted(by_parent.items(), key=lambda kv: -kv[1])[:15]:
        rel = folder.relative_to(root)
        print(f"   {n:>7,}  {rel}")
    if len(by_parent) > 15:
        print(f"   ... et {len(by_parent) - 15} autres")

    # 2) Tentative de détection open/closed par le nom des dossiers
    open_dirs, closed_dirs = [], []
    for folder in by_parent:
        name = folder.name.lower()
        if any(h in name for h in OPEN_HINTS):
            open_dirs.append(folder)
        elif any(h in name for h in CLOSED_HINTS):
            closed_dirs.append(folder)

    print("\nDétection automatique des classes par nom de dossier :")
    if open_dirs and closed_dirs:
        n_open = sum(by_parent[d] for d in open_dirs)
        n_closed = sum(by_parent[d] for d in closed_dirs)
        print(f"   OUVERT/ÉVEILLÉ  : {n_open:,} images dans {[str(d.relative_to(root)) for d in open_dirs]}")
        print(f"   FERMÉ/ENDORMI   : {n_closed:,} images dans {[str(d.relative_to(root)) for d in closed_dirs]}")
        if any("awake" in d.name.lower() or "sleepy" in d.name.lower() for d in open_dirs + closed_dirs):
            print("\n   ⚠ Noms 'awake'/'sleepy' détectés : VÉRIFIE le readme.md du dataset avant")
            print("     de continuer — confirme que awake=œil ouvert et sleepy=œil fermé.")
        print("\n   ✓ Structure reconnue. Ce dataset semble pré-découpé en test/train/val —")
        print("     on ignore volontairement ce découpage (probablement pas fait par sujet,")
        print("     donc en fuite de données potentielle) et on regroupe tout. Lance :\n")
        print(f"   python3 merge_dataset.py --root \"{root}\" \\")
        print(f"       --open {' '.join(str(d) for d in open_dirs)} \\")
        print(f"       --closed {' '.join(str(d) for d in closed_dirs)}")
    else:
        print("   ✗ Aucun nom de dossier ne contient 'open'/'closed'/'0'/'1'.")

        # Convention MRL brute : sSSSS_IIIII_G_R_L_E_S.png où
        #   G=genre E=lunettes L=lumière R=reflet et surtout
        #   E (5e champ, index 4) code l'état de l'œil : 0=fermé 1=ouvert.
        # Réf. mrl.cs.vsb.cz : "subject ID; image ID; gender; glasses;
        # eye state; reflections; lighting conditions; sensor id"
        mrl_like = [p for p in all_imgs if len(p.stem.split("_")) >= 5]
        if len(mrl_like) / len(all_imgs) > 0.9:
            print("   → Motif reconnu : convention MRL brute (état codé DANS le nom,")
            print("     pas dans le dossier). Champ 5 (index 4) : 0=fermé, 1=ouvert.\n")
            counts = Counter(p.stem.split("_")[4] for p in mrl_like if len(p.stem.split("_")) > 4)
            print(f"   Répartition du champ état observée : {dict(counts)}")
            print("\n   ✓ Commande de réorganisation (symlinks, ne duplique pas les 345 Mo) :\n")
            print("   mkdir -p train/data/open train/data/closed")
            print("   python3 - <<'PYEOF'")
            print("from pathlib import Path")
            print(f"src = Path('{root}')")
            print("dst_open, dst_closed = Path('train/data/open'), Path('train/data/closed')")
            print("dst_open.mkdir(parents=True, exist_ok=True)")
            print("dst_closed.mkdir(parents=True, exist_ok=True)")
            print("n = {'0': 0, '1': 0}")
            print("for p in src.rglob('*'):")
            print("    if p.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp'}:")
            print("        continue")
            print("    fields = p.stem.split('_')")
            print("    if len(fields) < 5:")
            print("        continue")
            print("    state = fields[4]")
            print("    dst = dst_open if state == '1' else dst_closed if state == '0' else None")
            print("    if dst is None:")
            print("        continue")
            print("    (dst / p.name).symlink_to(p.resolve())")
            print("    n[state] = n.get(state, 0) + 1")
            print("print('symlinks créés :', n)")
            print("PYEOF")
        else:
            print("   Regarde la liste ci-dessus et dis-moi quels dossiers correspondent")
            print("   à quelle classe, ou partage 10 noms de fichiers complets — le motif")
            print("   ne correspond ni à un tri par dossier ni à la convention MRL standard.")
            sample = [p.name for p in all_imgs[:20]]
            print(f"\n   Exemples de noms de fichiers : {sample[:8]}")
            parts_count = Counter(len(p.stem.split("_")) for p in all_imgs)
            print(f"   Nombre de champs séparés par '_' dans les noms : {dict(parts_count)}")

    # 3) Reconnaissance du sujet, indépendamment de la découverte ci-dessus
    matched = [p for p in all_imgs if SUBJECT_RE.search(p.name)]
    pct = 100 * len(matched) / len(all_imgs)
    print(f"\nIdentifiant de sujet reconnu (regex actuel) : {len(matched):,}/{len(all_imgs):,} ({pct:.0f}%)")
    if pct > 90:
        print("   ✓ --subject-regex par défaut fonctionne, rien à changer.")
    elif matched:
        subjects = defaultdict(int)
        for p in matched:
            subjects[SUBJECT_RE.search(p.name).group(1)] += 1
        print(f"   ⚠ Partiel seulement. {len(subjects)} sujets détectés parmi les images reconnues.")
        print("   Vérifie les noms non reconnus ci-dessous :")
        unmatched_sample = [p.name for p in all_imgs if not SUBJECT_RE.search(p.name)][:5]
        print(f"   {unmatched_sample}")
    else:
        print("   ✗ Aucun sujet reconnu. Partage 10 noms de fichiers réels pour ajuster le regex.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train/data_raw")
