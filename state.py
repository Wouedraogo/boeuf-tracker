"""
state.py
--------
État global partagé entre le thread de détection et les routes Flask.
Inclut un JSON encoder numpy-safe.
"""
import threading
import numpy as np
from flask.json.provider import DefaultJSONProvider


class NumpyJSONProvider(DefaultJSONProvider):
    """Sérialise les types numpy vers JSON sans planter."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


# Palette de 10 couleurs (BGR pour OpenCV)
PALETTE = [
    (16, 185, 129),    # vert
    (59, 130, 246),    # bleu
    (245, 158, 11),    # orange
    (236, 72, 153),    # rose
    (139, 92, 246),    # violet
    (34, 211, 238),    # cyan
    (250, 204, 21),    # jaune
    (248, 113, 113),   # rouge clair
    (52, 211, 153),    # vert clair
    (96, 165, 250),    # bleu clair
]


def color_for_name(name: str):
    """Couleur stable par nom (hash polynomial -> index palette)."""
    h = 0
    for c in name:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return PALETTE[h % len(PALETTE)]


STATE = {
    "frame_jpg": None,
    "frame_lock": threading.Lock(),
    "fps": 0.0,
    "device": "cpu",
    "started_at": None,
    "frame_count": 0,
    "active_animals": [],
    "events": [],
    "behavior": [],
    "track_history": {},
    "source": "",
    "source_label": "",
    "current_source_path": None,
    "desired_source": None,
    "desired_device": None,
    "desired_imgsz": None,        # nouvelle demande de resolution YOLO
    "desired_embed_every": None,  # nouvelle frequence d'embedding
    "desired_threshold": None,    # nouveau seuil cosine
    "desired_conf": None,         # nouvelle confiance YOLO
    "yolo_model_current": "",     # modele YOLO actif
    # Reglages d'inference par defaut. Cf. app.py pour la justification des
    # valeurs : 640/0.40 ne detectait qu'une fraction d'un troupeau serre.
    "imgsz_current": 1280,
    "embed_every_current": 10,
    "threshold_current": 0.70,
    "conf_current": 0.25,
    "ui_dir": "web/public",       # repertoire de l'UI statique (sert sans Bun)
    "models_available": [
        # CoreML/ANE (Apple Silicon, le plus rapide — imgsz FIGE a l'export).
        # On expose deux variantes par variante d'archi (small et medium), l'UI
        # switche automatiquement selon le cran de resolution choisi
        # (640 -> -640.mlpackage, 1280 -> -1280.mlpackage).
        # Medium (m) = ~15 pts mAP au-dessus du small sur COCO, meilleure
        # segmentation en scene barn/troupeau serre. ~2x plus lent que small.
        "yolo26m-seg-640.mlpackage",
        "yolo26m-seg-1280.mlpackage",
        "yolo26s-seg-640.mlpackage",
        "yolo26s-seg-1280.mlpackage",
        # YOLO26 MLX (Metal GPU Apple Silicon)
        "yolo26s-seg.safetensors",
        # YOLO26 PyTorch
        "yolo26m-seg.pt",
        "yolo26s-seg.pt",
        # YOLO11 standard
        "yolo11n-seg.pt", "yolo11s-seg.pt",
        "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt",
    ],
}


def reset_for_new_source():
    """À appeler quand la source change (webcam ↔ vidéo)."""
    with STATE["frame_lock"]:
        STATE["active_animals"] = []
        STATE["behavior"] = []
    STATE["track_history"].clear()
    STATE["events"].insert(0, "TRACKER reset")
    STATE["events"] = STATE["events"][:30]