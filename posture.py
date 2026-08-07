"""
posture.py
----------
Version epuree — la posture (couche/debout/tete-basse) est desormais
implicite dans les classes du R(2+1)D-18 (`resting-lying`, `resting-standing`,
`grazing`...), voir behavior_video.py.

Ce module ne contient plus qu'un helper de geometrie de boites utilise en
amont du Re-ID pour ecarter les crops trop occlus par un congenere :

  - `overlap_fractions(boxes)` : fraction de la surface de chaque boite qui
    est recouverte par une autre boite de la frame.

Historique : ce module contenait auparavant `MaskHeadAnalyzer`,
`HeadMotionTracker` et `HeadState` — signaux de posture derives de la
segmentation, qui alimentaient l'ancien classifieur d'action heuristique.
Ces mecanismes ont ete retires en meme temps que behavior.BehaviorAnalyzer
et posture_model.PostureModel.
"""
from __future__ import annotations

import numpy as np


def overlap_fractions(boxes) -> np.ndarray:
    """Pour chaque boite, la fraction de sa surface recouverte par une autre.

    On mesure `intersection / aire_de_la_boite`, pas l'IoU : ce qui compte
    n'est pas la ressemblance entre deux boites mais la part de l'animal
    qui est masquee. Un petit veau entierement cache derriere une adulte
    a une IoU faible mais une fraction de recouvrement proche de 1.

    Utilise pour ne pas mettre a jour l'empreinte de Re-ID a partir d'une
    imagette polluee (derive de l'empreinte stockee dans la DB).
    """
    arr = np.asarray(boxes, dtype=np.float32)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    if n == 1:
        return out

    areas = np.maximum(arr[:, 2] - arr[:, 0], 0) * np.maximum(arr[:, 3] - arr[:, 1], 0)
    for i in range(n):
        if areas[i] <= 0:
            continue
        x1 = np.maximum(arr[i, 0], arr[:, 0])
        y1 = np.maximum(arr[i, 1], arr[:, 1])
        x2 = np.minimum(arr[i, 2], arr[:, 2])
        y2 = np.minimum(arr[i, 3], arr[:, 3])
        inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        inter[i] = 0.0  # ne pas se comparer a soi-meme
        out[i] = float(inter.max()) / float(areas[i])
    return out
