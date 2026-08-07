"""
behavior.py
-----------
Version epuree — le comportement (pature, marche, couche, rumine...) est
desormais fourni entierement par le R(2+1)D-18 (voir behavior_video.py).

Ce fichier ne conserve que ce qui n'est PAS classification d'action :
  - EventJournal    : file d'evenements haut niveau (arrivee/depart/alertes)
  - IsolationMonitor: alerte "bovin isole du troupeau" (signal spatial, pas
                      lie a l'action individuelle)
  - NUMBA_OK        : flag utilise ailleurs pour l'accelerer si dispo

Historique : ce module contenait auparavant BehaviorClassifier / BehaviorAnalyzer
/ BehaviorTracker / TransitionMonitor et une pile de seuils manuels
(speed, aspect, immobile_dur, head_down_flags, lying_flags...). Ces
heuristiques etaient la source des faux positifs (bovin couche etiquete
"pature") et exigeaient une re-calibration a chaque tournage — elles ont
ete retirees en meme temps que posture.py/posture_model.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

try:
    from numba import njit as _njit  # noqa: F401 (importe pour usage externe)
    NUMBA_OK = True
except Exception:  # pragma: no cover
    NUMBA_OK = False


# ─────────────────────────────────────────────────────────────
#  Journal d'evenements
# ─────────────────────────────────────────────────────────────
@dataclass
class EventJournal:
    """File circulaire d'evenements haut niveau.

    Le buffer (STATE['events']) est fourni de l'exterieur pour rester
    compatible avec /api/stats qui expose STATE en JSON.
    """
    events_ref: list
    max_events: int = 40

    def push(self, kind: str, text: str, name: str | None = None) -> None:
        self.events_ref.insert(0, {
            "kind": kind, "text": text, "name": name, "ts": time.time(),
        })
        del self.events_ref[self.max_events:]


# ─────────────────────────────────────────────────────────────
#  Detection d'isolement (bovin loin du centroide du troupeau)
# ─────────────────────────────────────────────────────────────
@dataclass
class IsolationMonitor:
    journal: EventJournal
    dist_frac_threshold: float = 0.35
    min_duration_s: float = 6.0
    _isolated_since: dict[str, float] = field(default_factory=dict)
    _flagged: set[str] = field(default_factory=set)

    def update(
        self,
        boxes: np.ndarray | list,
        track_ids: Iterable[int],
        frame_shape: tuple[int, int, int],
        track_names: dict[int, str],
        now: float | None = None,
    ) -> None:
        now = now if now is not None else time.time()
        if boxes is None or len(boxes) < 3:
            self._isolated_since.clear()
            return

        fh, fw = frame_shape[:2]
        diag = (fw * fw + fh * fh) ** 0.5
        centers = np.array([
            ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes
        ])
        troop = centers.mean(axis=0)

        active_names: set[str] = set()
        for c, tid in zip(centers, track_ids):
            name = track_names.get(int(tid))
            if not name or name == "?":
                continue
            active_names.add(name)
            dist_frac = float(np.linalg.norm(c - troop)) / diag
            if dist_frac > self.dist_frac_threshold:
                self._isolated_since.setdefault(name, now)
                if (name not in self._flagged
                        and now - self._isolated_since[name] > self.min_duration_s):
                    self._flagged.add(name)
                    self.journal.push("alert", f"{name} s'est isole du troupeau", name=name)
            else:
                self._isolated_since.pop(name, None)
                self._flagged.discard(name)

        for gone in list(self._isolated_since.keys()):
            if gone not in active_names:
                self._isolated_since.pop(gone, None)
                self._flagged.discard(gone)
