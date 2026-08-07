"""
training/train_cbvd5.py
-----------------------
Entraine R(2+1)D-18 sur le dataset PUBLIC **CBVD-5** (Fandaoerji et al., 2024).

Contexte : le checkpoint behavior_video.pt distribue avec le projet a ete entraine
sur le dataset CVB (paturage exterieur). En scene barn/auge, il produit des faux
"couche" / "marche" parce que les scenes barn sont hors distribution. CBVD-5
capture 96 h de filmage BARN de 107 vaches avec 5 comportements : standing,
lying, foraging (mange a l'auge), rumination, drinking — c'est PILE le domaine
qui nous manque.

Source dataset : https://www.kaggle.com/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset
Paper         : https://www.nature.com/articles/s41598-024-65953-x  (Nature 2024)

Usage :
    # 1) Recuperer le dataset (voir training/download_cbvd5.sh)
    # 2) Lancer l'entrainement (GPU/CUDA fortement recommande, MPS OK mais lent)
    .venv/bin/python training/train_cbvd5.py \
        --data external/cbvd5 --epochs 20 --batch 8

Sortie : behavior_video.pt  (ecrase l'ancien — les 5 nouvelles classes sont
         mappees en FR par _VIDEO_BEHAVIOR_FR dans processor.py).

Notes structure du dataset :
- Kaggle publie CBVD-5 dans plusieurs layouts selon les mirrors. Ce script
  detecte automatiquement soit :
    a) <root>/<class_name>/*.mp4                (le plus courant)
    b) <root>/videos/<class_name>/*.mp4
    c) <root>/<class_name>/<clip_id>/*.jpg      (frames pre-extraites)
- Les noms de classes attendus (canoniques) : standing, lying, foraging,
  rumination, drinking. Aliases toleres via CLASS_ALIASES ci-dessous.
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


# ─────────────────────────────────────────────────────────────
#  Taxonomie CBVD-5 → nom canonique en anglais kebab-case.
#  On garde le format anglais pour rester coherent avec les autres
#  checkpoints (processor.py::_VIDEO_BEHAVIOR_FR fait la traduction FR).
# ─────────────────────────────────────────────────────────────
CLASS_NAMES: list[str] = [
    "standing", "lying", "foraging", "rumination", "drinking",
]

# Aliases lower-case eventuellement rencontres dans les dossiers Kaggle.
CLASS_ALIASES: dict[str, str] = {
    "stand": "standing", "standing": "standing",
    "lie": "lying", "lying": "lying", "lyingdown": "lying", "lying_down": "lying",
    "forage": "foraging", "foraging": "foraging",
    "ruminate": "rumination", "rumination": "rumination", "ruminating": "rumination",
    "drink": "drinking", "drinking": "drinking",
}

CLIP_LEN, SIZE = 16, 112
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


def _normalize_class(name: str) -> str | None:
    """Renvoie le nom canonique du dossier, ou None si non reconnu."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return CLASS_ALIASES.get(key)


def _iter_class_dirs(root: str) -> Iterable[tuple[str, str]]:
    """Yield (class_canonical, path_to_class_dir) pour chaque layout supporte."""
    # Layout a) <root>/<class>/
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            c = _normalize_class(d)
            if c is not None:
                yield c, p
    # Layout b) <root>/videos/<class>/
    videos_root = os.path.join(root, "videos")
    if os.path.isdir(videos_root):
        for d in sorted(os.listdir(videos_root)):
            p = os.path.join(videos_root, d)
            if os.path.isdir(p):
                c = _normalize_class(d)
                if c is not None:
                    yield c, p


def _sample_video_clip(video_path: str) -> np.ndarray | None:
    """Extrait CLIP_LEN frames equi-espacees d'un .mp4 → (T, H, W, 3) RGB uint8."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None
    idxs = np.linspace(0, max(0, total - 1), CLIP_LEN).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            cap.release()
            return None
        fr = cv2.cvtColor(cv2.resize(fr, (SIZE, SIZE)), cv2.COLOR_BGR2RGB)
        frames.append(fr)
    cap.release()
    return np.stack(frames)  # (T,H,W,3)


def _sample_frames_clip(frames: list[str]) -> np.ndarray | None:
    """Meme chose depuis une liste de .jpg deja extraits."""
    if len(frames) < CLIP_LEN:
        return None
    idxs = np.linspace(0, len(frames) - 1, CLIP_LEN).astype(int)
    buf = []
    for i in idxs:
        img = cv2.imread(frames[i])
        if img is None:
            return None
        buf.append(cv2.cvtColor(cv2.resize(img, (SIZE, SIZE)), cv2.COLOR_BGR2RGB))
    return np.stack(buf)


class CBVD5Clips(Dataset):
    """Dataset unifie : chaque item = (path_or_frames, label_int, is_video)."""

    def __init__(self, root: str):
        self.items: list[tuple[list[str] | str, int, bool]] = []
        class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
        per_class = {c: 0 for c in CLASS_NAMES}
        for cname, cdir in _iter_class_dirs(root):
            label = class_to_idx[cname]
            # Videos .mp4 directement dans le dossier de classe
            for v in sorted(glob.glob(os.path.join(cdir, "*.mp4"))):
                self.items.append((v, label, True))
                per_class[cname] += 1
            for v in sorted(glob.glob(os.path.join(cdir, "*.avi"))):
                self.items.append((v, label, True))
                per_class[cname] += 1
            # Sous-dossiers de frames pre-extraites
            for sub in sorted(glob.glob(os.path.join(cdir, "*"))):
                if os.path.isdir(sub):
                    jpgs = sorted(glob.glob(os.path.join(sub, "*.jpg")))
                    if len(jpgs) >= CLIP_LEN:
                        self.items.append((jpgs, label, False))
                        per_class[cname] += 1
        if not self.items:
            raise SystemExit(
                f"Aucun clip trouve sous {root}. Layouts attendus :\n"
                f"  <root>/<class>/*.mp4        OU\n"
                f"  <root>/videos/<class>/*.mp4 OU\n"
                f"  <root>/<class>/<id>/*.jpg\n"
                f"Classes attendues (aliases toleres) : {list(CLASS_ALIASES.keys())}"
            )
        print(f"Corpus : {len(self.items)} clips")
        for c, n in per_class.items():
            print(f"  {c:12s} : {n}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        src, label, is_video = self.items[i]
        clip = _sample_video_clip(src) if is_video else _sample_frames_clip(src)
        if clip is None:
            # Fallback : renvoie un clip noir plutot que de crasher le DataLoader
            clip = np.zeros((CLIP_LEN, SIZE, SIZE, 3), dtype=np.uint8)
        x = torch.from_numpy(clip).float().div_(255.0)   # (T,H,W,3)
        x = x.permute(3, 0, 1, 2).contiguous()            # (3,T,H,W)
        x = (x - MEAN) / STD
        return x, label


def build_model(num_classes: int) -> nn.Module:
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="Racine du dataset CBVD-5 extrait (voir download_cbvd5.sh).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--out", default="behavior_video.pt",
                   help="Ecrit dans la racine du projet (ecrase le .pt actuel).")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device : {device} · classes : {CLASS_NAMES}")

    ds = CBVD5Clips(args.data)
    n_val = max(1, int(args.val_split * len(ds)))
    tr, va = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    vl = DataLoader(va, batch_size=args.batch, num_workers=args.workers)

    model = build_model(len(CLASS_NAMES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    best_acc = 0.0
    for ep in range(args.epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); lossf(model(x), y).backward(); opt.step()
        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                pred = model(x.to(device)).argmax(1).cpu()
                correct += (pred == y).sum().item(); total += y.numel()
        acc = correct / max(1, total)
        print(f"epoch {ep+1}/{args.epochs}  val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "state_dict": model.cpu().state_dict(),
                "class_names": CLASS_NAMES,
                "clip_len": CLIP_LEN, "size": SIZE,
                "mean": MEAN, "std": STD, "arch": "r2plus1d_18",
                "dataset": "CBVD-5", "val_acc": acc,
            }, args.out)
            model.to(device)
            print(f"  ↑ meilleur, sauve dans {args.out}")

    print(f"\nFini. Meilleure val_acc : {best_acc:.3f}  ({args.out})")


if __name__ == "__main__":
    main()
