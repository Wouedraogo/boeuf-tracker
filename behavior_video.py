"""
behavior_video.py
-----------------
Inférence du comportement VIDÉO (R(2+1)D-18, torchvision) branchée sur le
pipeline existant. S'utilise en plus de behavior.py : pour chaque bœuf suivi,
on accumule ses crops sur une fenêtre de 16 frames et on interroge le modèle
périodiquement → comportement dynamique (grazing, ruminating, walking…).

Charge le checkpoint de training/train_behavior_video.py (behavior_video.pt).

Dégradé propre : si le checkpoint est absent, `available=False` et update()
renvoie None (le pipeline continue avec posture/vitesse).
"""
from __future__ import annotations

import os
from collections import deque, defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

from console import info, ok, warn


class VideoBehavior:
    def __init__(self,
                 ckpt: str = "behavior_video.pt",
                 device: str | None = None,
                 every: int = 8):
        self.available = False
        self.every = every
        self._counter: dict[int, int] = defaultdict(int)
        self._buffers: dict[int, deque] = {}
        self._labels: dict[int, str] = {}
        self._confs: dict[int, float] = {}

        if not os.path.exists(ckpt):
            warn(f"[VideoBehavior] checkpoint absent ({ckpt}) — comportement vidéo désactivé")
            return

        if device is None:
            # CUDA (Colab) si dispo, sinon CPU. On évite MPS : la conv 3D
            # plante sur Metal (command buffer errors).
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        info(f"[VideoBehavior] Chargement de {ckpt} sur {device}...")
        blob = torch.load(ckpt, map_location=device)
        self.class_names: list[str] = blob["class_names"]
        self.clip_len: int = blob["clip_len"]
        self.size: int = blob["size"]
        self.mean = blob["mean"].to(device).view(1, 3, 1, 1, 1)
        self.std = blob["std"].to(device).view(1, 3, 1, 1, 1)

        model = r2plus1d_18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.class_names))
        model.load_state_dict(blob["state_dict"])
        self.model = model.to(device).eval()
        self.available = True
        ok(f"[VideoBehavior] Prêt — {len(self.class_names)} comportements : {self.class_names}")

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        img = cv2.cvtColor(cv2.resize(crop_bgr, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).float().div_(255.0).permute(2, 0, 1)  # (3,H,W)

    @staticmethod
    def apply_mask(crop_bgr: np.ndarray, mask_bool: np.ndarray | None) -> np.ndarray:
        """Blackline le fond hors du masque de segmentation.

        Le modele a ete entraine sur des scenes paturage. En scene barn, la
        bbox brute est dominee par des barreaux/foin/beton hors distribution,
        ce qui produit des faux "walking"/"couche". Isoler l'animal via son
        masque de segmentation ramene le crop dans une distribution proche du
        training set (silhouette animale sur fond neutre).
        """
        if mask_bool is None or crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr
        if mask_bool.shape != crop_bgr.shape[:2]:
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8), (crop_bgr.shape[1], crop_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        out = np.zeros_like(crop_bgr)
        out[mask_bool] = crop_bgr[mask_bool]
        return out

    def update(
        self,
        track_id: int,
        crop_bgr: np.ndarray,
        mask_bool: np.ndarray | None = None,
    ) -> tuple[str, float] | None:
        """Ajoute un crop au buffer du bœuf et renvoie (label, confidence).

        `mask_bool` (optionnel) : masque de segmentation en coordonnees crop.
        Si fourni, le fond est blackliste avant preprocessing → focus sur
        l'animal, ignore le contexte (barreaux, foin, beton, congeneres).
        """
        if not self.available or crop_bgr is None or crop_bgr.size == 0:
            return None

        crop_bgr = self.apply_mask(crop_bgr, mask_bool)
        buf = self._buffers.setdefault(track_id, deque(maxlen=self.clip_len))
        buf.append(self._preprocess(crop_bgr))
        self._counter[track_id] += 1

        if len(buf) == self.clip_len and self._counter[track_id] % self.every == 0:
            clip = torch.stack(list(buf), dim=1).unsqueeze(0).to(self.device)  # (1,3,T,H,W)
            clip = (clip - self.mean) / self.std
            with torch.no_grad():
                logits = self.model(clip)
                probs = torch.softmax(logits, dim=1)[0]
                conf, idx = torch.max(probs, dim=0)
            self._labels[track_id] = self.class_names[int(idx.item())]
            self._confs[track_id] = float(conf.item())

        label = self._labels.get(track_id)
        if label is None:
            return None
        return label, self._confs.get(track_id, 0.0)

    def forget(self, track_id: int) -> None:
        self._buffers.pop(track_id, None)
        self._labels.pop(track_id, None)
        self._confs.pop(track_id, None)
        self._counter.pop(track_id, None)
