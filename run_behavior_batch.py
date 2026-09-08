"""
run_behavior_batch.py
---------------------
Inference COMPORTEMENT en batch, 100 % headless (aucune fenetre, aucun serveur
web) sur un dossier de videos.

Deux sources de comportement, fusionnees :
  1. YOLO26-seg fine-tune CBVD-5 (boeuf_cbvd5_seg/weights/best.pt) : detection
     + segmentation + classe d'action directe (standing/lying/eating/...).
  2. R(2+1)D-18 (behavior_video.pt) : classification video sur 16 frames de
     crop masque par bovin suivi (grazing/ruminating/walking/...).
Le R(2+1)D gagne des qu'il depasse le seuil de confiance, sinon on retombe sur
la classe YOLO de la frame.

Sortie, dans --out :
  - <video>_behavior.mp4  : video annotee (boites + masque + action FR + HUD)
  - montage_behavior.mp4  : toutes les videos annotees concatenees (--montage)
  - behavior_report.json  : stats par video et par piste

Usage :
    .venv/bin/python run_behavior_batch.py --input samples2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from behavior_video import VideoBehavior

# Table de traduction EN -> FR, alignee sur processor._VIDEO_BEHAVIOR_FR pour que
# les libelles de la video batch soient identiques a ceux du dashboard temps reel.
# Recopiee ici plutot qu'importee : importer processor instancie tout le pipeline
# (re-ID, breed, analytics) dont ce script batch n'a pas besoin.
BEHAVIOR_FR = {
    # YOLO26 CBVD-5
    "standing": "debout", "lying": "couche", "eating": "mange",
    "drinking": "boit", "walking": "marche", "running": "court", "other": "",
    # R(2+1)D CVB (8 classes)
    "grazing": "pature", "ruminating-standing": "rumine debout",
    "ruminating-lying": "rumine couche", "resting-standing": "debout",
    "resting-lying": "couche", "foraging": "mange", "rumination": "rumine",
}

# Palette stable par piste (BGR).
_PALETTE = [
    (66, 135, 245), (76, 201, 132), (245, 176, 66), (219, 84, 97),
    (168, 100, 230), (66, 224, 245), (240, 120, 190), (140, 200, 80),
]


def color_for(tid: int) -> tuple[int, int, int]:
    return _PALETTE[tid % len(_PALETTE)]


def masks_to_frame(result, shape) -> dict[int, np.ndarray]:
    """Rasterise les polygones de segmentation en masques plein cadre.

    On passe par `masks.xy` (deja en coordonnees image d'origine) plutot que
    par `masks.data` (resolution reseau) : pas de rescaling a faire soi-meme.
    """
    out: dict[int, np.ndarray] = {}
    if result.masks is None:
        return out
    h, w = shape[:2]
    for i, poly in enumerate(result.masks.xy):
        if poly is None or len(poly) < 3:
            continue
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [poly.astype(np.int32)], 1)
        out[i] = m.astype(bool)
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU de chaque boite de `a` (N,4) contre chaque boite de `b` (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def actions_by_box(action_model, frame, args):
    """Classe d'action CBVD-5 par boite : renvoie (boites, [(label_fr, conf)])."""
    if action_model is None:
        return np.zeros((0, 4), np.float32), []
    r = action_model.predict(frame, imgsz=args.imgsz, conf=args.conf_action,
                             verbose=False, device=args.device)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4), np.float32), []
    boxes = r.boxes.xyxy.cpu().numpy()
    labels = [
        (BEHAVIOR_FR.get(action_model.names[int(c)], action_model.names[int(c)]), float(cf))
        for c, cf in zip(r.boxes.cls.cpu().numpy().astype(int),
                         r.boxes.conf.cpu().numpy())
    ]
    return boxes, labels


def draw_overlay(frame, x1, y1, x2, y2, mask, color, label):
    if mask is not None:
        tint = np.zeros_like(frame)
        tint[mask] = color
        cv2.addWeighted(tint, 0.35, frame, 1.0, 0, dst=frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if not label:
        return
    fs, ft = 0.45, 1
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
    ly1 = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, ly1), (x1 + tw + 8, ly1 + th + 8), color, -1)
    cv2.putText(frame, label, (x1 + 4, ly1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), ft, cv2.LINE_AA)


def draw_hud(frame, lines):
    h = 18 * len(lines) + 10
    cv2.rectangle(frame, (0, 0), (frame.shape[1], h), (24, 24, 24), -1)
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)


def to_h264(src: str, dst: str) -> bool:
    """Re-encode en H.264 yuv420p : mp4v d'OpenCV ne lit pas dans un navigateur."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p", dst],
            check=True,
        )
        os.remove(src)
        return True
    except Exception as e:
        print(f"  [warn] re-encode H.264 impossible ({e}) — on garde {src}")
        return False


COW_CLASS_IDS = None  # renseigne dans main() si le modele est un COCO generique


def process_video(path, model, action_model, behavior, args, out_dir):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  [skip] illisible : {path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stem = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
    tmp = os.path.join(out_dir, f".{stem}_raw.mp4")
    final = os.path.join(out_dir, f"{stem}_behavior.mp4")
    stride = max(1, args.stride)
    out_fps = fps / stride  # duree reelle preservee
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    # Le tracker ultralytics est persistant par instance de modele : on jette
    # le predictor entre deux videos pour ne pas heriter des pistes de la
    # precedente. (Vider predictor.trackers ne marche pas : le callback
    # on_predict_postprocess_end y accede sans garde.)
    model.predictor = None

    per_track = defaultdict(Counter)   # tid -> Counter(action_fr -> frames)
    frame_idx = 0      # frames ecrites (= inferees)
    read_idx = 0       # frames lues dans la source
    t0 = time.time()

    while True:
        okr, frame = cap.read()
        if not okr:
            break
        read_idx += 1
        if (read_idx - 1) % stride:
            continue

        res = model.track(frame, persist=True, conf=args.conf, iou=0.5,
                          imgsz=args.imgsz, verbose=False, device=args.device,
                          tracker="bytetrack.yaml")[0]

        annotated = frame.copy()
        counts = Counter()
        act_boxes, act_labels = actions_by_box(action_model, frame, args)

        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            conf = res.boxes.conf.cpu().numpy()
            ids = (res.boxes.id.int().cpu().numpy()
                   if res.boxes.id is not None
                   else np.arange(len(xyxy)))
            masks = masks_to_frame(res, frame.shape)
            # Appariement piste -> boite du modele d'action (meilleur IoU).
            ious = iou_matrix(xyxy, act_boxes)

            for i, (box, tid, c, cf) in enumerate(zip(xyxy, ids, cls, conf)):
                # Sur un modele COCO, on garde 'cow' mais aussi 'horse'/'sheep' :
                # COCO confond regulierement un bovin de dos ou lointain avec
                # l'un des deux (meme choix que detector.py).
                if COW_CLASS_IDS is not None and int(c) not in COW_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 < 16 or y2 - y1 < 16:
                    continue
                tid = int(tid)

                mask_full = masks.get(i)
                crop = frame[y1:y2, x1:x2]
                mask_crop = mask_full[y1:y2, x1:x2] if mask_full is not None else None

                # --- Source 1 : R(2+1)D sur crop masque (fenetre 16 frames) ---
                vb = behavior.update(tid, crop, mask_crop) if behavior and behavior.available else None
                # --- Source 2 : classe d'action YOLO de la frame courante ---
                # --- Source 2 : classe d'action CBVD-5 (barn) appariee par IoU ---
                act = None
                if ious.shape[1]:
                    j = int(np.argmax(ious[i]))
                    if ious[i, j] >= args.iou_match:
                        act = act_labels[j]
                # --- Source 3 : classe du modele de detection s'il porte des actions ---
                own_raw = model.names.get(int(c), "")
                own = ((BEHAVIOR_FR.get(own_raw, own_raw), float(cf))
                       if not COW_CLASS_IDS else None)

                # Priorite : CBVD-5 (entraine en stabulation, per-frame) > R(2+1)D
                # (temporel mais entraine sur du paturage, derive ici) > classe propre.
                if act and act[0]:
                    action, acted_conf = act
                elif vb is not None and vb[1] >= args.min_conf:
                    action, acted_conf = BEHAVIOR_FR.get(vb[0], vb[0]), vb[1]
                elif own and own[0]:
                    action, acted_conf = own
                else:
                    action, acted_conf = "", 0.0

                if action:
                    counts[action] += 1
                    per_track[tid][action] += 1

                label = f"#{tid} {action} {acted_conf:.2f}" if action else f"#{tid}"
                draw_overlay(annotated, x1, y1, x2, y2, mask_full,
                             color_for(tid), label)

        elapsed = time.time() - t0
        hud = [
            f"{os.path.basename(path)}   frame {read_idx}/{total}   "
            f"{(frame_idx + 1) / max(elapsed, 1e-6):.1f} fps",
            "bovins: %d   |   %s" % (
                sum(counts.values()),
                "  ".join(f"{k}={v}" for k, v in counts.most_common(5)) or "-",
            ),
        ]
        draw_hud(annotated, hud)
        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"    {read_idx}/{total}  ({frame_idx / max(elapsed, 1e-6):.1f} fps inf)",
                  flush=True)

    cap.release()
    writer.release()
    to_h264(tmp, final)

    dur = time.time() - t0
    summary = {
        "video": os.path.basename(path),
        "output": os.path.basename(final),
        "frames_source": read_idx,
        "frames_inferees": frame_idx,
        "stride": stride,
        "fps_source": round(fps, 2),
        "fps_sortie": round(out_fps, 2),
        "processing_fps": round(frame_idx / max(dur, 1e-6), 2),
        "seconds": round(dur, 1),
        "tracks": len(per_track),
        "per_track": {
            str(tid): {
                "dominant": cnt.most_common(1)[0][0] if cnt else None,
                "frames_par_action": dict(cnt.most_common()),
                "secondes_par_action": {k: round(v / out_fps, 1) for k, v in cnt.most_common()},
            }
            for tid, cnt in sorted(per_track.items())
        },
    }
    print(f"  -> {final}  ({frame_idx} frames, {len(per_track)} pistes, "
          f"{summary['processing_fps']} fps)")
    return summary


def build_montage(videos, out_path, size=(832, 464)):
    """Concatene les videos annotees, mises a la meme resolution (padding)."""
    if len(videos) < 2:
        return None
    lst = out_path + ".txt"
    scaled = []
    for i, v in enumerate(videos):
        s = f"{out_path}.part{i}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", v, "-vf",
             f"scale={size[0]}:{size[1]}:force_original_aspect_ratio=decrease,"
             f"pad={size[0]}:{size[1]}:(ow-iw)/2:(oh-ih)/2,setsar=1",
             "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p", s],
            check=True,
        )
        scaled.append(s)
    with open(lst, "w") as f:
        for s in scaled:
            f.write(f"file '{os.path.abspath(s)}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", lst, "-c", "copy", out_path], check=True)
    for s in scaled:
        os.remove(s)
    os.remove(lst)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Inference comportement en batch, headless.")
    p.add_argument("--input", default="samples2", help="Dossier de videos ou fichier unique.")
    p.add_argument("--out", default="outputs_behavior", help="Dossier de sortie.")
    p.add_argument("--model", default="yolo26s-seg.pt",
                   help="YOLO segmentation pour detection+tracking. Le COCO-seg "
                        "generique rappelle ~3x plus de bovins que le fine-tune "
                        "CBVD-5 sur les scenes de stabulation (mesure sur samples2). "
                        "Passer boeuf_cbvd5_seg/weights/best.pt pour la taxonomie "
                        "d'action YOLO directe.")
    p.add_argument("--action-model", default="boeuf_cbvd5_lr001/weights/best.pt",
                   help="2e YOLO, fine-tune CBVD-5 (standing/lying/eating/...), "
                        "utilise UNIQUEMENT pour la classe d'action : ses boites "
                        "sont appariees par IoU aux pistes du modele de detection. "
                        "'' pour desactiver.")
    p.add_argument("--conf-action", type=float, default=0.10,
                   help="Confiance min du modele d'action. Basse par defaut : ses "
                        "boites ne servent qu'a classer une piste deja detectee "
                        "par le modele principal, donc un faux positif isole est "
                        "sans effet alors qu'un rappel faible laisse des pistes "
                        "sans action (0.25 -> 50 %% des pistes etiquetees, "
                        "0.10 -> 67 %% sur samples2).")
    p.add_argument("--iou-match", type=float, default=0.40,
                   help="IoU minimal pour rattacher une boite d'action a une piste.")
    p.add_argument("--behavior-ckpt", default="behavior_video.pt",
                   help="Checkpoint R(2+1)D-18. '' pour desactiver.")
    p.add_argument("--imgsz", type=int, default=832)
    p.add_argument("--stride", type=int, default=2,
                   help="Inference 1 frame sur N. La source est a 60 fps : "
                        "stride=2 conserve la duree reelle (sortie a 30 fps) "
                        "pour 2x moins de calcul.")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--every", type=int, default=8,
                   help="Periode (frames) d'inference R(2+1)D par piste.")
    p.add_argument("--min-conf", type=float, default=0.55,
                   help="Softmax minimal du R(2+1)D. Il a ete entraine sur des "
                        "scenes de paturage (CVB) et derive en stabulation : "
                        "sous 0.55 ses sorties sont proches du hasard (1/8=0.125), "
                        "on prefere ne rien afficher.")
    p.add_argument("--device", default=None, help="'mps', 'cpu', 'cuda:0'. Auto par defaut.")
    p.add_argument("--montage", action="store_true", help="Produire aussi la video concatenee.")
    p.add_argument("--limit", type=int, default=0, help="Ne traiter que les N premieres videos.")
    args = p.parse_args()

    if args.device is None:
        args.device = ("cuda:0" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")

    if os.path.isdir(args.input):
        videos = sorted(
            g for ext in ("mp4", "mov", "MP4", "MOV", "avi")
            for g in glob.glob(os.path.join(args.input, f"*.{ext}"))
        )
    else:
        videos = [args.input]
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        sys.exit(f"Aucune video dans {args.input}")

    os.makedirs(args.out, exist_ok=True)
    print(f"[Batch] {len(videos)} videos | device={args.device} | imgsz={args.imgsz}")
    print(f"[Modele] detection+action : {args.model}")

    model = YOLO(args.model)
    names = model.names
    global COW_CLASS_IDS
    if "cow" in names.values():
        COW_CLASS_IDS = {i for i, n in names.items() if n in ("cow", "horse", "sheep")}
        print(f"[Modele COCO] classes bovines retenues : {sorted(COW_CLASS_IDS)}")
    else:
        COW_CLASS_IDS = None
        print(f"[Classes YOLO] {names}")

    action_model = None
    if args.action_model and os.path.exists(args.action_model):
        action_model = YOLO(args.action_model)
        print(f"[Modele action] {args.action_model} -> {action_model.names}")
    elif args.action_model:
        print(f"[warn] modele d'action introuvable : {args.action_model} — ignore")

    behavior = None
    if args.behavior_ckpt:
        behavior = VideoBehavior(ckpt=args.behavior_ckpt, every=args.every)

    report, produced = [], []
    t0 = time.time()
    for i, v in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {os.path.basename(v)}")
        s = process_video(v, model, action_model, behavior, args, args.out)
        if s:
            report.append(s)
            produced.append(os.path.join(args.out, s["output"]))

    montage = None
    if args.montage and len(produced) > 1:
        print("\n[Montage] concatenation...")
        montage = build_montage(produced, os.path.join(args.out, "montage_behavior.mp4"))
        print(f"  -> {montage}")

    # Agregat global : combien de frames-bovin par action, toutes videos confondues.
    glob_actions = Counter()
    for s in report:
        for t in s["per_track"].values():
            for k, v in t["frames_par_action"].items():
                glob_actions[k] += v

    rep_path = os.path.join(args.out, "behavior_report.json")
    with open(rep_path, "w") as f:
        json.dump({
            "modele_detection": args.model,
            "modele_action_frame": args.action_model,
            "modele_action_video": args.behavior_ckpt,
            "device": args.device,
            "imgsz": args.imgsz,
            "duree_totale_s": round(time.time() - t0, 1),
            "actions_globales_frames": dict(glob_actions.most_common()),
            "montage": os.path.basename(montage) if montage else None,
            "videos": report,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] {len(report)} videos annotees en {time.time() - t0:.0f}s -> {args.out}/")
    print(f"[Rapport] {rep_path}")
    print(f"[Actions cumulees] {dict(glob_actions.most_common())}")


if __name__ == "__main__":
    main()
