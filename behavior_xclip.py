"""
behavior_xclip.py
-----------------
Classifieur de comportement zero-shot via X-CLIP (Microsoft, 2022).

Contrat identique a behavior_video.VideoBehavior :
  - .available : bool
  - .update(track_id, crop_bgr, mask_bool=None) -> (label_key, softmax_conf) | None
  - .forget(track_id)

Difference clef : au lieu d'un modele entraine sur une taxonomie figee
(les 8 classes de behavior_video.pt), X-CLIP compare l'embedding video du
clip a des embeddings texte que TU redefinis en Python. Ajouter un
comportement = ajouter une phrase, pas re-entrainer. Le label_key est la
clef du dict de prompts (ex. "pature", "mange_auge", "couche"...).

Poids : microsoft/xclip-base-patch32  (~1.1 GB au premier telechargement).
Devine : CPU par defaut sur Mac (MPS supporte mais transformers peut
prompt-cache mal), CUDA si dispo.
"""
from __future__ import annotations

import os
from collections import defaultdict, deque

import cv2
import numpy as np
import torch

from console import info, ok, warn


# ─────────────────────────────────────────────────────────────
#  Taxonomie des prompts. Cle = label FR affiche, valeur = phrase EN
#  (X-CLIP est trainee sur ~10M paires video/texte EN).
#  Pour affiner : reformule la phrase, teste, itere. Aucun re-training.
# ─────────────────────────────────────────────────────────────
DEFAULT_PROMPTS: dict[str, str] = {
    "pature":         "a cow grazing on grass in a field",
    "marche":         "a cow walking",
    "rumine debout":  "a cow chewing cud while standing still",
    "rumine couche":  "a cow chewing cud while lying down on the ground",
    "debout":         "a cow standing still doing nothing",
    "couche":         "a cow lying down on the ground",
    "boit":           "a cow drinking water from a trough",
    "court":          "a cow running fast",
    # NOUVEAUX prompts pour scenes barn — le point que R(2+1)D ratait,
    # ajoutes ici sans dataset ni retrain grace au zero-shot.
    "mange auge":     "a cow eating hay at a feeding trough or feed bunk in a barn",
    "immobile barn":  "a cow standing motionless in a barn stall",
}


class VideoBehaviorXClip:
    """Interface drop-in de VideoBehavior utilisant X-CLIP zero-shot."""

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        model_name: str = "microsoft/xclip-base-patch32",
        device: str | None = None,
        every: int = 16,
        clip_len: int = 8,   # X-CLIP-base-patch32 a ete pre-entraine avec 8 frames
    ):
        self.available = False
        self.every = every
        self.clip_len = clip_len
        self._counter: dict[int, int] = defaultdict(int)
        self._buffers: dict[int, deque] = {}
        self._labels: dict[int, str] = {}
        self._confs: dict[int, float] = {}

        try:
            from transformers import AutoProcessor, XCLIPModel  # type: ignore
        except Exception as e:
            warn(f"[XClip] transformers XCLIP indispo ({e}) — module desactive")
            return

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        info(f"[XClip] Chargement de {model_name} sur {device} (~1 GB au 1er run)...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = XCLIPModel.from_pretrained(model_name).to(device).eval()
        except Exception as e:
            warn(f"[XClip] echec chargement modele : {e}")
            return

        self.prompts = prompts or DEFAULT_PROMPTS
        self.labels: list[str] = list(self.prompts.keys())
        self.phrases: list[str] = list(self.prompts.values())

        # Pre-encode texte une fois pour toutes : c'est stable et coute cher
        # si recalcule par frame. On garde les features sur device.
        with torch.no_grad():
            text_inp = self.processor(
                text=self.phrases, return_tensors="pt", padding=True,
            ).to(device)
            text_feats = self.model.get_text_features(**text_inp)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        self._text_features = text_feats  # (K, D)

        self.available = True
        ok(f"[XClip] Pret — {len(self.labels)} comportements : {self.labels}")

    @staticmethod
    def apply_mask(crop_bgr: np.ndarray, mask_bool: np.ndarray | None) -> np.ndarray:
        """Noircit le fond hors du masque de segmentation (idem VideoBehavior)."""
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

    def _preprocess_frame(self, crop_bgr: np.ndarray) -> np.ndarray:
        """RGB uint8 (H,W,3) au format attendu par le processor."""
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    def update(
        self,
        track_id: int,
        crop_bgr: np.ndarray,
        mask_bool: np.ndarray | None = None,
    ) -> tuple[str, float] | None:
        if not self.available or crop_bgr is None or crop_bgr.size == 0:
            return None

        crop_bgr = self.apply_mask(crop_bgr, mask_bool)
        buf = self._buffers.setdefault(track_id, deque(maxlen=self.clip_len))
        buf.append(self._preprocess_frame(crop_bgr))
        self._counter[track_id] += 1

        # Inference tous les `every` appels, une fois le buffer plein.
        if len(buf) == self.clip_len and self._counter[track_id] % self.every == 0:
            frames = list(buf)  # list of (H,W,3) uint8 RGB
            with torch.no_grad():
                inputs = self.processor(
                    videos=[frames], return_tensors="pt",
                ).to(self.device)
                video_feats = self.model.get_video_features(**inputs)
                video_feats = video_feats / video_feats.norm(dim=-1, keepdim=True)
                # similarite cosinus texte-video → softmax
                logits = (video_feats @ self._text_features.T) * self.model.logit_scale.exp()
                probs = torch.softmax(logits, dim=-1)[0]
                conf, idx = torch.max(probs, dim=0)
            self._labels[track_id] = self.labels[int(idx.item())]
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
