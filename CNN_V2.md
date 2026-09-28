# FocusLens v2 — le CNN passe en première ligne

Ce lot corrige la chaîne de détection oculaire et inverse la hiérarchie entre
les deux sources : le réseau entraîné décide, la géométrie contrôle et supplée.

## Fichiers remplacés

| Fichier | Ce qui change |
|---|---|
| `src/config.py` | `use_cnn_eye: bool` devient `eye_source: "cnn" \| "auto" \| "ear"`. Ajout du seuil calibré, de l'EMA CNN et de la bande d'incertitude. |
| `src/temporal.py` | Accepte un signal d'ouverture CNN en plus de l'EAR. Seuils absolus côté CNN, relatifs côté EAR. Mesure le taux de désaccord entre sources. Chaque événement porte sa source. |
| `src/analyzer.py` | Le CNN pilote, l'EAR est calculé à chaque frame pour la calibration, le contrôle et le repli. Nouvelle méthode `source_summary()`. |
| `src/eye_cnn_infer.py` | Plus de pseudo-EAR. Chargement tolérant. Seuil, température et cadrage lus dans le checkpoint. Pondération des deux yeux par la qualité du patch. |
| `train/models/eye_cnn.py` | Entrée 64×64, ordre des classes corrigé, pooling adaptatif. |
| `train/train_eye_cnn.py` | Découpe par sujet, augmentation étendue, calibration de température, choix du seuil, métriques avec « fermé » comme classe positive. |
| `train/benchmark_ear_vs_cnn.py` | Nouveau. Compare les deux sources sur la même vidéo. |

## Les trois bugs corrigés

**L'ordre des classes.** `ImageFolder` trie alphabétiquement, donc `closed=0`
et `open=1`. La v1 déclarait l'inverse et lisait la colonne 1 comme la
probabilité « fermé ». Conséquence directe : la précision de 0,960 et le rappel
de 0,913 publiés dans `metrics.json` décrivaient la classe **ouvert**. Recalculé
depuis la matrice `[[398, 17], [39, 409]]`, le rappel réel sur œil fermé était
de 95,9 %, soit 4,1 % de fermetures ratées.

**La fuite de données.** `random_split` sur l'ensemble des images place des
frames voisines du même œil de la même personne dans train et dans test. Le
score mesurait la mémorisation. La v2 réserve des sujets entiers au test.
Attends-toi à une baisse : c'est le signe que le chiffre est devenu honnête.

**Le pseudo-EAR.** La v1 convertissait la probabilité du réseau en
`0.34 - p * 0.24` pour ne pas toucher à la couche temporelle. Un classifieur
sature à 0 ou 1, donc ce faux EAR était un créneau : la vitesse de fermeture
était toujours mesurée comme rapide, et **aucun micro-sommeil ne pouvait être
classé critique par la dynamique**. C'est corrigé par la calibration de
température plus le lissage EMA, qui rendent au signal une pente exploitable.

Vérifié sur une fermeture lente simulée de 1,5 seconde : la v2 classe
l'événement `critique` dans les deux modes, avec une vitesse de fermeture de
−0,44 en EAR et −0,67 en CNN, bien sous le seuil de 3,0.

## Réentraîner

```bash
pip install torch torchvision scikit-learn matplotlib pillow

python train/train_eye_cnn.py --data train/data --epochs 25
```
pip install kaggle
kaggle datasets download -d akashshingha850/mrl-eye-dataset
Expand-Archive -Path .\mrl-eye-dataset.zip -DestinationPath .\train\data_raw

Options utiles :

```bash
--target-recall 0.98    # rappel visé sur "fermé" pour choisir le seuil
--img-size 64           # 48 si la machine peine, 96 pour tester plus fin
--width 32              # largeur du premier bloc conv
--random-split          # reproduit le protocole v1, pour comparer les deux chiffres
```

Le dernier est intéressant pour ta soutenance : lance les deux, montre l'écart,
explique d'où il vient. C'est exactement le genre de rigueur qu'un jury cherche.

Le checkpoint contient désormais le seuil, la température, la taille d'entrée et
le padding du crop. L'inférence les reprend seule, il n'y a plus rien à
synchroniser à la main entre entraînement et production.

## Basculer FocusLens

Dans `src/config.py`, `eye_source` vaut `"cnn"` par défaut. Trois valeurs :

- `"cnn"` — le réseau décide. Si torch ou le checkpoint manquent, bascule
  automatique sur l'EAR avec un message, sans interrompre la session.
- `"auto"` — le réseau décide, sauf quand sa probabilité tombe dans la bande
  d'incertitude : ces frames-là sont tranchées par la géométrie.
- `"ear"` — géométrie seule, aucune dépendance torch.

`FocusLens.source_summary()` rend la part de frames décidées par chaque source
et le taux d'accord entre elles. À afficher dans le rapport : c'est la
traçabilité du système.

## Protocole de démonstration devant le jury

Tourne une vidéo de 90 secondes en simulant la conduite, avec des événements
que tu connais : deux clignements normaux, un clignement volontaire long mais
rapide, une fermeture lente de trois secondes, et un passage tête tournée.
Note les instants dans un CSV.

```csv
t_start,t_end
34.2,37.5
```

Puis :

```bash
python train/benchmark_ear_vs_cnn.py --video demo.mp4 --truth demo_truth.csv
```

Tu obtiens dans `bench/` :

- `signals.png` — les deux signaux superposés dans le temps, avec leurs seuils
  et une bande d'événements comparant vérité, EAR et CNN
- `report.md` — le tableau comparatif, rappel et fausses alarmes inclus
- `events.csv` — chaque événement des deux sources avec sa preuve chiffrée
- `summary.json` — les mêmes chiffres, exploitables ailleurs

La même vidéo, les deux sources, une figure. C'est la diapositive.
