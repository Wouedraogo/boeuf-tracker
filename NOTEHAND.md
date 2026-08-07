# NOTEHAND — Optimisations CoreML + corrections UI (2026-08-05)

## Ce qui a ete fait

### 1. Conversion CoreML pour le Neural Engine (ANE)

**YOLO26s-seg** converti en `.mlpackage` float16 (20 Mo) pour le Neural Engine Apple.

- Benchmark : **~37 FPS** (CoreML/ANE) vs ~23 FPS (PyTorch MPS) = **+60%**
- Entree fixe 1280x1280 (l'ANE exige des shapes fixes)
- Conversion : `model.export(format="coreml", half=True, nms=True, imgsz=1280)` via Ultralytics

**MegaDescriptor** teste en CoreML mais abandonne : seulement +8% de gain d'inference pour 60s de chargement au demarrage (vs 4s en PyTorch MPS). Reste sur PyTorch MPS.

**R(2+1)D-18** (comportement video) : conv3D ne tourne PAS sur l'ANE (fallback CPU confirme par Apple). Reste sur CPU. MPS crashe sur Metal avec les conv 3D.

### 2. Integration dans le code

**processor.py** — Auto-detection au demarrage :
- Si `yolo26s-seg.mlpackage` existe et qu'on est sur macOS → utilise CoreML/ANE
- Sinon fallback sur MLX (Metal GPU) si dispo, puis PyTorch MPS
- Priorite : CoreML/ANE > MLX/Metal > PyTorch/MPS > CPU

**reid.py** — `MegaDescriptorCoreML` ajoute (disponible mais pas active par defaut). Peut etre injecte via `CattleReID(deep_extractor=MegaDescriptorCoreML(...))`.

**state.py** — Liste des modeles mise a jour : CoreML en premier, YOLO26 .pt/.safetensors, puis YOLO11 (sans les variantes non-seg inutiles).

**detector.py** — Support reload CoreML dans le hot-swap de modele.

### 3. Correctifs labels video

**Accents (`??`)** : `cv2.putText` ne supporte pas Unicode — les caracteres accentues (pature, couche, rue) s'affichaient comme `p??ture`. Corrige avec `unicodedata.normalize('NFD')` qui retire les accents avant l'affichage.

**Stabilite des labels** : les comportements changeaient a chaque frame (flickering). Ajout d'un lissage temporel (`LABEL_STICKY_FRAMES=5`) — un label ne change que s'il persiste 5 frames consecutives.

### 4. Corrections UI (precedentes)

**Video feed** : le polling JPEG (`/video_feed`) ne demarre plus au chargement de la page. Il attend qu'une source soit active (evite les GET inutiles dans la console).

**Modeles** : la liste deroulante dans les settings affiche maintenant le `.mlpackage` CoreML en premier.

### 5. Fichiers generes (non versiones)

- `yolo26s-seg.mlpackage` (20 Mo) — a regenerer avec : `python -c "from ultralytics import YOLO; YOLO('yolo26s-seg.pt').export(format='coreml', half=True, nms=True, imgsz=1280)"`

### 6. .gitignore

Ajouts :
- `external/` (datasets Roboflow/Kaggle, 31 Go)
- `training/_img_excluded/`, `training/_video_crops_backup/`
- `*.mlpackage`, `*.mlmodelc`, `*.zip`

## Ce qui n'a PAS ete change

- `behavior_video.pt` reste sur CPU (conv3D incompatible ANE/MPS)
- MegaDescriptor reste sur PyTorch MPS (CoreML pas rentable)
- `dev.sh` inchange — lance toujours `python app.py --mlx` + `bun run src/server.ts`
- Quand le modele CoreML est actif, le selecteur imgsz n'a pas d'effet (entree fixe 1280x1280 dans le modele compile)

## Comment lancer

```bash
./dev.sh
```

Ouvre http://localhost:8000, choisis une video dans "Source". Le modele CoreML est auto-detecte si `yolo26s-seg.mlpackage` est present a la racine.
