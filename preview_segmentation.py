#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from reid import CattleReID
from database import EmbeddingDatabase
from names import make_name_generator
from state import color_for_name

def run_preview(video_path=None):
    if not video_path:
        # Default test video
        video_path = '/Volumes/Untitled/downloads/IMG_3641.MOV'
        if not os.path.exists(video_path):
            video_path = 'samples/IMG_3641.MOV'
            
    if not os.path.exists(video_path):
        print(f'Erreur: fichier {video_path} introuvable.')
        return

    print(f'=== Lancement du Streaming de Segmentation sur {os.path.basename(video_path)} ===')
    print('Commandes : Appuyez sur [Q] ou [ESC] pour quitter, [ESPACE] pour pause.')
    
    seg_model = YOLO('yolo26s-seg.pt' if os.path.exists('yolo26s-seg.pt') else 'yolo11s-seg.pt')
    reid = CattleReID(model_name='hf-hub:BVRA/MegaDescriptor-T-224', device='cpu')
    db = EmbeddingDatabase(path='temp_prev.pkl', reid_engine=reid)
    name_gen = make_name_generator(db)
    track_id_to_name = {}

    cap = cv2.VideoCapture(video_path)
    f_idx = 0
    paused = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            f_idx += 1
            h, w = frame.shape[:2]
            
            results = seg_model.predict(frame, conf=0.25, imgsz=832, classes=[19], verbose=False)[0]
            annotated = frame.copy()
            mask_overlay = np.zeros_like(frame, dtype=np.uint8)

            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                masks = results.masks.data.cpu().numpy() if results.masks is not None else None
                
                for det_idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if (x2 - x1) < 20 or (y2 - y1) < 20:
                        continue
                    
                    crop = frame[y1:y2, x1:x2]
                    if det_idx not in track_id_to_name or f_idx % 25 == 0:
                        emb = reid.get_embedding(crop)
                        if emb is not None:
                            matched, _ = db.match(emb, threshold=0.65)
                            k = matched or f'Boeuf_{len(db.animals)+1:03d}'
                            if matched is None:
                                db.add(k, emb)
                                name_gen.__init__(db.animals)
                            track_id_to_name[det_idx] = k

                    k = track_id_to_name.get(det_idx, f'Boeuf_{det_idx+1:03d}')
                    nom = name_gen.get(k)
                    color = color_for_name(k)
                    
                    aspect = (x2 - x1) / max((y2 - y1), 1)
                    norm_y2 = y2 / max(h, 1)
                    beh = 'mange' if norm_y2 > 0.85 and (y2 - y1) > h * 0.4 else ('couche' if aspect > 1.35 else 'debout')
                    
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if masks is not None and det_idx < len(masks):
                        mask_raw = masks[det_idx]
                        mask_resized = cv2.resize(mask_raw.astype(np.uint8), (w, h))
                        mask_overlay[mask_resized > 0.5] = (np.array(color) * 0.35).astype(np.uint8)
                        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(annotated, contours, -1, color, 3)
                        M = cv2.moments(mask_resized)
                        if M['m00'] > 0:
                            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])

                    label = f'{nom} • {beh}'
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    px1, py1 = max(0, cx - tw // 2 - 8), max(0, min(cy, y1 + 30) - th - 6)
                    cv2.rectangle(annotated, (px1, py1), (px1 + tw + 16, py1 + th + 12), color, -1)
                    cv2.putText(annotated, label, (px1 + 8, py1 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            blended = cv2.addWeighted(annotated, 1.0, mask_overlay, 0.7, 0)
            cv2.imshow('Boeuf Tracker — Streaming Segmentation Contours', cv2.resize(blended, (1080, 608)))

        key = cv2.waitKey(20) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists('temp_prev.pkl'):
        os.remove('temp_prev.pkl')

if __name__ == '__main__':
    v = sys.argv[1] if len(sys.argv) > 1 else None
    run_preview(v)
