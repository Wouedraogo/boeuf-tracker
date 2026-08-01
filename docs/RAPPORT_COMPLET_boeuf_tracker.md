# Boeuf Tracker — Rapport Technique Complet

**Système de Surveillance et d'Identification de Bovins par Vision par Ordinateur**

---

| | |
|---|---|
| **Projet** | Boeuf Tracker v2.0.0 |
| **Type** | PFE Individuel — Projet de Fin d'Études |
| **Cours** | GEI1052 — Génie Électrique |
| **Session** | Hiver 2026 |
| **Auteur** | Ismael Gansonre |
| **Date du rapport** | 22 juillet 2026 |
| **Dépôt Git** | `boeuf-tracker` (55 commits, 6 jours de développement actif) |
| **Plateformes** | macOS (principal), Windows, Linux |

---

## Table des matières

1. [Résumé exécutif](#1-résumé-exécutif)
2. [Contexte et problematic](#2-contexte-et-problématique)
3. [Objectifs du projet](#3-objectifs-du-projet)
4. [État de l'art](#4-état-de-lart)
5. [Architecture globale du système](#5-architecture-globale-du-système)
6. [Détection YOLO et segmentation d'instances](#6-détection-yolo-et-segmentation-dinstances)
7. [Suivi multi-objets (tracking)](#7-suivi-multi-objets-tracking)
8. [Ré-identification (Re-ID) avec DINOv2](#8-ré-identification-re-id-avec-dinov2)
9. [Classification de race avec SigLIP-2](#9-classification-de-race-avec-siglip-2)
10. [Génération de noms et persistance d'identité](#10-génération-de-noms-et-persistance-didentité)
11. [Pipeline de traitement vidéo](#11-pipeline-de-traitement-vidéo)
12. [Classification des comportements](#12-classification-des-comportements)
13. [Analyse de données et tableau de bord](#13-analyse-de-données-et-tableau-de-bord)
14. [Carte de chaleur spatiale (heatmap)](#14-carte-de-chaleur-spatiale-heatmap)
15. [Serveur Flask et API REST](#15-serveur-flask-et-api-rest)
16. [Interface utilisateur web](#16-interface-utilisateur-web)
17. [Application desktop Tauri](#17-application-desktop-tauri)
18. [Optimisations de performance](#18-optimisations-de-performance)
19. [Robustesse et gestion d'erreurs](#19-robustesse-et-gestion-derreurs)
20. [Évolution chronologique du projet](#20-évolution-chronologique-du-projet)
21. [Résultats et métriques](#21-résultats-et-métriques)
22. [Limites connues](#22-limites-connues)
23. [Perspectives et améliorations futures](#23-perspectives-et-améliorations-futures)
24. [Guide d'installation et de déploiement](#24-guide-dinstallation-et-de-déploiement)
25. [Structure du code source](#25-structure-du-code-source)
26. [Glossaire technique](#26-glossaire-technique)
27. [Références](#27-références)

---

## 1. Résumé exécutif

**Boeuf Tracker** est un système complet de surveillance d'élevage bovin reposant sur la vision par ordinateur et l'intelligence artificielle. Conçu dans le cadre d'un Projet de Fin d'Études en Génie Électrique (GEI1052, Hiver 2026), il permet de détecter, identifier individuellement, classer par race, et analyser spatialement les bovins présents dans un flux vidéo en temps réel.

Le système combine quatre technologies d'IA de pointe :

- **YOLOv26 sur MLX (Metal)** pour la détection et segmentation d'instances en temps réel (~12 ms par frame)
- **DINOv2-small** pour la ré-identification (Re-ID) cross-vidéo via embeddings de 464 dimensions (~5 ms par batch asynchrone)
- **SigLIP-2 So400m** pour la classification zero-shot de 10 races bovines françaises (~22 ms, nouveau bovin seulement)
- **Numba JIT** pour la classification de comportements (couché, pâture, marche, etc.)

L'application atteint **23-26 FPS** sur Apple M1 Pro, largement au-delà de l'objectif de 15 FPS, avec une précision de détection d'environ **92 %** sur les vidéos de test.

Le projet inclut une interface web complète (dark/light theme), un tableau de bord analytique avec carte de chaleur spatiale, des profils individuels par bovin, et un wrapper desktop **Tauri v2** pour un déploiement cross-platform (macOS, Windows, Linux).

La persistance d'identité est garantie par un système de base de données par vidéo (`.pkl`) combiné à un compteur global persistant (`names_counter.json`), assurant que chaque bovin reçoit un nom unique qui ne sera jamais réutilisé, même après réinitialisation.

**Versions successives des modèles de classification de race :**
1. Heuristique HSV (robe coat color)
2. CLIP ViT-B/32 (zero-shot)
3. **SigLIP-2 So400m** (état de l'art 2025, 1136M paramètres) — version finale

---

## 2. Contexte et problématique

### 2.1 Le défi de la surveillance d'élevage

L'élevage bovin moderne représente un secteur économique majeur, particulièrement en France où l'on compte environ **19 millions de bovins** répartis sur quelque 180 000 exploitations. La surveillance manuelle de tels cheptels est :

- **Coûteuse** en main-d'œuvre (estimation : 2-3 heures/ jour/ exploitation pour le seul comptage)
- **Erreur-prone** : le comptage visuel d'un troupeau de 200 têtes a une marge d'erreur de 5-10 %
- **Limitée** : impossible à réaliser en continu (24/7), sur de grandes surfaces, ou de nuit
- **Subjective** : l'identification individuelle par numéro d'oreille/douane nécessite une proximité physique

### 2.2 Limites des solutions existantes

Les systèmes RFID et colliers connectés (ex. : Allflex SenseHub, CowManager) dominent le marché professionnel. Ils présentent cependant des contraintes :

| Solution | Coût par tête | Limitation principale |
|---|---|---|
| RFID passif (boucle d'oreille) | 3-5 € | Lecture manuelle ou portail obligatoire, pas de localisation |
| RFID actif + capteurs | 80-150 € | Investissement infrastructure (antennes, passerelles) |
| Collier connecté IoT | 100-300 € | Batterie limitée (2-3 ans), confort animal |
| Caméras thermiques | > 10 000 € | Coût élevé, portée limitée |

### 2.3 L'approche vision par ordinateur

La vision par ordinateur (CV) offre une alternative **non-invasive, sans capteur sur l'animal, et 24/7**. Cependant, identifier *individuellement* des bovins visuellement est un défi majeur car :

- Les bovins d'une même race se ressemblent énormément (variabilité inter-individuelle faible)
- Les conditions d'éclairage varient (jour/nuit, ombres, intérieur/ extérieur)
- Les animaux se déplacent, s'occlusionnent mutuellement
- Le suivi doit être maintenu sur de longues durées et entre vidéos distinctes

C'est précisément ce problème — **ré-identifier un même bovin à travers des vidéos différentes** — que Boeuf Tracker résout.

### 2.4 Cadre académique

Ce projet s'inscrit dans le cours **GEI1052** du Baccalauréat en Génie Électrique de l'Université (session Hiver 2026). Les objectifs pédagogiques couvrent :

- Intégration de modèles d'IA pré-entraînés dans un système opérationnel
- Optimisation de performance pour des contraintes temps-réel
- Architecture logicielle multi-couches (IA + API + UI + desktop)
- Déploiement cross-platform

---

## 3. Objectifs du projet

### 3.1 Objectif principal

> Concevoir un système temps-réel capable de détecter, identifier individuellement et classer par race les bovins dans un flux vidéo, sans capteur sur l'animal.

### 3.2 Objectifs spécifiques initialement fixés

1. **Détection et segmentation** des bovins par image (YOLO + instance segmentation)
2. **Suivi intra-vidéo** (tracking) pour attribuer un ID temporaire par frame
3. **Ré-identification cross-vidéo** : reconnaître qu'un bovin vu dans la vidéo A est le même que dans la vidéo B
4. **Classification de race** automatique (10 races françaises principales)
5. **Interface utilisateur** web avec visualisation temps-réel
6. **Performance ≥ 15 FPS** sur matériel grand public

### 3.3 Objectifs additionnels réalisés (au-delà du cahier des charges)

7. **Classification de comportements** (7 états : couché, pâture, boit, immobile, marche, court, rué) via Numba JIT
8. **Tableau de bord analytique** : KPIs, graphiques FPS/races/activités, carte de chaleur spatiale, timeline
9. **Profils individuels** par bovin avec historique croisé-vidéos
10. **Application desktop** Tauri cross-platform (macOS/Windows/Linux)
11. **Noms propres persistants** : chaque bovin reçoit un nom unique jamais réutilisé (ex. : "Marguerite", "Aurelius")
12. **Optimisation Apple Silicon** : MLX/Metal pour YOLOv26, FP16 pour DINOv2
13. **Hot-reload** des paramètres (modèle, résolution, seuils) sans redémarrage
14. **Auto-récupération** en cas de crash de la source vidéo
15. **Documentation complète** : doc technique, rapport final, 6 diagrammes, 3 présentations PPTX

---

## 4. État de l'art

### 4.1 Détection d'objets : évolution des architectures

| Génération | Modèles emblématiques | Vitesse | Précision | Caractéristique |
|---|---|---|---|---|
| Two-stage | Faster R-CNN (2015) | Lente | Très haute | Régions candidates puis classification |
| Single-stage v1 | SSD (2016), YOLOv1-3 | Rapide | Bonne | Détection dense en 1 passe |
| Single-stage v2 | YOLOv4-8 (2020-2023) | Très rapide | Haute | Anchor-free, CSP, PANet |
| Single-stage v3 | YOLOv9-11 (2024) | Ultra rapide | Très haute | Programmable gradient information |
| MLX-native | **YOLOv26 (2025)** | Extrême | Très haute | Optimisé Apple Silicon Metal |

**Choix du projet** : YOLOv11s-seg (PyTorch) en fallback, **YOLOv26s-seg sur MLX** en production sur macOS — ce dernier étant environ **2,6× plus rapide** que YOLO11 sur PyTorch/MPS.

### 4.2 Ré-identification visuelle (Re-ID)

La ré-identification consiste à reconnaître une même entité à travers des caméras ou vidéos distinctes. Trois familles d'approches existent :

1. **Méthodes supervisées** : entraînement d'un réseau triplet/siamese sur un jeu de données annoté (ex. : vehicle Re-ID, person Re-ID). Nécessite un dataset annoté de bovins — qui n'existe pas publiquement.

2. **Transfer learning** : utiliser un modèle pré-entraîné sur une tâche générique (ImageNet) et extraire ses embeddings. C'est l'approche choisie avec **DINOv2-small** (384-dim).

3. **Self-supervised learning** : DINOv2 est précisément un modèle auto-supervisé (DINO = self-DIstillation with NO labels), pré-entraîné sur 142 millions d'images. Il produit des embeddings remarquablement génériques qui capturent l'apparence sémantique sans annotation — idéal pour une tâche comme la nôtre sans données annotées de bovins.

**Pourquoi DINOv2 plutôt que CLIP ?** DINOv2 produit des features d'apparence pure (non alignées avec le langage), ce qui le rend supérieur pour la discrimination visuelle fine. CLIP, à l'inverse, aligne image et texte — utile pour la classification de race mais moins optimal pour la distinction d'individus proches.

### 4.3 Classification zero-shot de races

La classification zero-shot permet de reconnaître des classes **sans aucune image d'entraînement** de ces classes. Deux approches dominent :

- **CLIP** (OpenAI, 2021) : alignement image-texte sur 400M paires. ViT-B/32.
- **SigLIP** (Google, 2023) : remplace la contrastive loss softmax par une sigmoid loss, permettant un entraînement plus scalable.
- **SigLIP-2** (Google, 2025) : état de l'art actuel. La variante **So400m** (400M paramètres visuels, 1136M au total) atteint **84,1 % top-1 sur ImageNet** en zero-shot.

**Choix du projet** : SigLIP-2 So400m (`google/siglip2-so400m-patch16-384`) — le modèle le plus performant disponible en 2025 pour la classification zero-shot fine-grained. Aucune image d'entraînement de race n'étant disponible, le zero-shot est la seule option viable.

### 4.4 Outils de surveillance d'élevage académiques

Les travaux académiques existants sur la surveillance bovine par vision sont principalement :

- **Comptage** (relativement trivial avec YOLO)
- **Détection de comportement** (mastication, déplacement)
- **Estimation de poids** par morphologie

Très peu de travaux traitent de la **ré-identification individuelle** de bovins par vision — c'est ce qui distingue ce projet. Les approches existantes reposent généralement sur des modèles entraînés sur mesure (nécessitant des milliers d'annotations), alors que Boeuf Tracker contourne ce problème via DINOv2 pré-entraîné.

---

## 5. Architecture globale du système

### 5.1 Vue d'ensemble à deux processus

Le système suit une architecture **client-serveur à deux processus** :

```
┌────────────────────────────────────────────────────────────────┐
│                    APPLICATION DESKTOP TAURI                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Fenêtre native (WKWebView / WebView2) 1280×800         │  │
│  │  ├─ Splash screen (splash.html) au démarrage             │  │
│  │  └─ Navigation vers http://127.0.0.1:8100 quand prêt     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                     spawn (std::process::Command)               │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  WORKER PYTHON (Flask, 127.0.0.1:8100)                  │  │
│  │                                                          │  │
│  │  Thread 1 : détection vidéo (daemon)                    │  │
│  │    ├─ YOLO (MLX/Metal ou PyTorch)                       │  │
│  │    ├─ DINOv2 Re-ID (via ReIDWorker async)               │  │
│  │    ├─ SigLIP-2 breed classification                     │  │
│  │    └─ Numba behavior classification                     │  │
│  │                                                          │  │
│  │  Thread 2 : serveur Flask                                │  │
│  │    ├─ /api/stats, /api/dashboard, /api/heatmap...       │  │
│  │    ├─ /video_feed (JPEG polling)                         │  │
│  │    └─ sert aussi web/public/ (HTML/JS/CSS)              │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

Le worker Python est **auto-suffisant** : il sert à la fois l'API REST et l'interface web. Tauri n'est qu'un wrapper qui lance et supervise ce worker.

### 5.2 Choix de conception : worker HTTP plutôt qu'IPC

Contrairement à l'approche Tauri classique (IPC entre Rust et frontend via `invoke`), le projet communique entièrement via **HTTP localhost**. Avantages :

- **Indépendance** : le worker fonctionne sans Tauri (accessible via navigateur)
- **Debug facilité** : `curl localhost:8100/api/stats` à tout moment
- **Pas de schéma IPC** à maintenir
- **Frontend inchangé** entre mode web et mode desktop

### 5.3 Couches logicielles

| Couche | Technologie | Rôle |
|---|---|---|
| **Couche IA** | YOLOv26, DINOv2, SigLIP-2, Numba | Inférence ML |
| **Couche logique** | Python (processor.py, detector.py, reid.py...) | Orchestration pipeline |
| **Couche persistance** | Pickle (.pkl), JSON (counter, history) | Stockage embeddings/ données |
| **Couche API** | Flask 3.x | Serveur HTTP REST |
| **Couche UI** | HTML5, CSS3, JavaScript vanilla | Interface |
| **Couche desktop** | Tauri v2 (Rust) | Wrapper natif |
| **Couche optionnelle** | Bun + Hono | Proxy de développement |

### 5.4 Diagramme de flux de données

Le pipeline par frame suit ce flux séquentiel avec parties asynchrones :

```
Frame capturée (OpenCV)
     │
     ├─ [skip adaptatif si FPS < 20]
     ▼
Détection YOLO (~12 ms MLX)
     │ boxes + track_ids + masks + confs
     ▼
Collecte des crops (nouveaux tracks + re-embed dus)
     │
     ├───────────────┐ (asynchrone)
     ▼               ▼
DINOv2 Re-ID    Boucle principale continue
(~5 ms batch)   ├─ annotate_frame (~9 ms)
     │          ├─ analyze_behavior (Numba ~0.1 ms)
     │          ├─ db.match() (BLAS vectorisé ~0.1 ms)
     │          └─ JPEG encode off-thread (~3 ms)
     ▼
SigLIP-2 breed (uniquement NOUVEAU bovin, ~22 ms)
     │
     ▼
STATE["frame_jpg"] → servi par /video_feed
```

---

## 6. Détection YOLO et segmentation d'instances

### 6.1 Rôle de la détection

La détection est la **première étape** du pipeline. Pour chaque frame vidéo, YOLO identifie les bovins en :

- Localisant leur boîte englobante (bounding box `[x1, y1, x2, y2]`)
- Segmentant leur silhouette (mask pixel-level)
- Assignant un score de confiance (`conf ∈ [0,1]`)
- Maintenant un ID de suivi temporaire (`track_id`)

### 6.2 Backends de détection

Deux backends sont implémentés, sélectionnés automatiquement :

#### 6.2.1 Backend PyTorch (`CattleDetector`)

- **Modèle** : YOLOv11s-seg (Ultralytics)
- **Poids** : `yolo11s-seg.pt` (20,7 Mo)
- **Tracker** : ByteTrack (`tracker="bytetrack.yaml"`)
- **Device** : auto-sélection `cuda > mps > cpu`
- **FP16** : activé sur `cuda` et `mps` (conversion manuelle `.model.model.half()` car Ultralytics 8.4+ a déprécié le flag `half=`)
- **Warmup** : un passage sur image dummy 64×64 pour initialiser les kernels

#### 6.2.2 Backend MLX (`CattleDetectorMLX`) — production sur macOS

- **Modèle** : YOLOv26s-seg (yolo26mlx)
- **Poids** : `yolo26s-seg.safetensors` (46,3 Mo) ou `.pt` (23,5 Mo)
- **Device** : `"mlx"` (toujours Metal GPU)
- **FP16** : natif (MLX est FP16 by design)

**Pourquoi un tracker custom en MLX ?** La bibliothèque `yolo26mlx` expose `predict()` et `track()`, mais `TrackerManager.update()` **droit les masks** lors du tracking (bug documenté). Le projet implémente donc un `_SimpleIoUTracker` maison et utilise `predict()` + tracking manuel.

### 6.3 Le tracker IoU maison (`_SimpleIoUTracker`)

Un tracker multi-objet minimaliste basé sur l'IoU (Intersection over Union) :

**Paramètres :**
- `iou_threshold = 0.3` — IoU minimum pour associer une détection à un track existant
- `max_lost = 30` — nombre de frames avant de supprimer un track perdu

**Algorithme (`update()`) :**
1. Vieillir tous les tracks existants (`lost += 1`)
2. Pour chaque nouvelle détection :
   - Calculer l'IoU avec tous les tracks existants
   - Sélectionner le track avec le meilleur IoU (≥ 0.3)
   - Si match : réutiliser son ID, reset `lost = 0`
   - Sinon : allouer un nouvel ID monotone
3. Supprimer les tracks avec `lost > max_lost`

**Limitation assumée** : ce tracker est adéquat pour des bovins lents, mais perd les IDs en cas d'occlusion prolongée ou de mouvement rapide. C'est compensé par le Re-ID DINOv2 qui réattribue l'identité stable indépendamment du track_id.

### 6.4 Compatibilité backend-agnostic

Un point crucial de l'architecture : le code consommateur (processor.py) utilise des appels de style PyTorch :

```python
result.boxes.xyxy.cpu().numpy()
result.boxes.id.int().cpu().numpy()
result.masks.data.cpu().numpy()
```

Pour que ceci fonctionne aussi avec les résultats MLX (qui sont des arrays numpy), une couche de **shims de compatibilité** est implémentée :

| Shim | Rôle |
|---|---|
| `_NumpyWrapper` | Fait qu'un array numpy a `.cpu()` qui retourne un `_CpuView` |
| `_CpuView` | Supporte `.numpy()`, `.int()`, `.cpu()` comme un tensor torch |
| `_MLXBoxes` | Reproduit `.xyxy`, `.id`, `.conf`, `.cls` |
| `_MLXMasks` | Reproduit `.data` |
| `_MLXResult` | Wrapper de résultat complet compatible Ultralytics |

Ainsi, **aucun branchement** `if backend == "mlx"` n'est nécessaire dans le pipeline principal.

### 6.5 Classe COCO utilisée

YOLO est entraîné sur COCO (80 classes). La classe **vache** est l'ID **19** (`COW_CLASS_ID = 19`). La détection filtre donc :

```python
model.track(frame, classes=[19], tracker="bytetrack.yaml", persist=True)
```

Le paramètre `persist=True` permet au tracker de maintenir l'état entre les appels (IDs cohérents intra-vidéo).

### 6.6 Sécurité MLX : le lock

Un bug de `yolo26mlx` provoque un SIGABRT si plusieurs threads appellent Metal simultanément. La classe `CattleDetectorMLX` sérialise donc tous les appels via :

```python
self._mlx_lock = threading.Lock()

def detect(self, ...):
    with self._mlx_lock:
        return self._detect_inner(...)
```

C'est un compromis : la détection ne peut pas paralléliser, mais le Re-ID (sur PyTorch/MPS) tourne en parallèle dans `ReIDWorker`, donc le parallélisme est préservé entre YOLO et DINOv2.

---

## 7. Suivi multi-objets (tracking)

### 7.1 Distinction tracking vs Re-ID

| Concept | Portée | Durée | Mécanisme |
|---|---|---|---|
| **Tracking** (ByteTrack / IoU) | Intra-vidéo | Frame à frame | Continuité spatiale (IoU) |
| **Re-ID** (DINOv2) | Cross-vidéo | Permanent | Empreinte visuelle (embedding) |

Le tracking est **éphémère** : si un bovin sort du champ puis revient, ByteTrack perd son ID et lui en attribue un nouveau. Le Re-ID, lui, reconnaît l'empreinte visuelle et restaure l'identité permanente (`Boeuf_042`).

### 7.2 Le double système d'identité

Chaque bovin détecté possède deux identités simultanées :

1. **`track_id`** (YOLO) — entier temporaire, peut changer d'une frame à l'autre
2. **`Boeuf_NNN`** (Re-ID) — clé permanente stockée en DB, jamais réutilisée

La correspondance est maintenue dans `track_id_to_name`, un dictionnaire mis à jour à chaque frame :

```python
track_id_to_name = {}  # {7: "Boeuf_042", 12: "Boeuf_043", ...}
```

Quand un nouveau `track_id` apparaît, on calcule son embedding DINOv2 et on interroge la DB. Si match → on récupère son `Boeuf_NNN`. Sinon → nouveau bovin, on en crée un.

### 7.3 Anti-double-comptage sur boucle vidéo

Un problème subtil : quand une vidéo fichier arrive à la fin et **reboucle** (rewind), tous les bovins réapparaissent avec de nouveaux `track_id`. Sans protection, le système les compterait comme nouveaux.

**Solution implémentée** : détection de rewind via `CAP_PROP_POS_FRAMES` :

```python
current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
if current_pos < last_pos:  # regression = rewind
    loop_detected_at_frame = frame_count
    track_id_to_name.clear()  # forcera le re-Re-ID
```

Pendant les **60 frames** suivant un rewind (`loop_grace_frames`), le seuil de Re-ID est abaissé de 0.70 à 0.45 (`loop_threshold`) pour être plus permissif et récupérer les identités existantes.

---

## 8. Ré-identification (Re-ID) avec DINOv2

### 8.1 Principe

La ré-identification repose sur la notion d'**embedding** : une représentation vectorielle dense de l'apparence visuelle d'un bovin, telle que deux images du même animal produisent des vecteurs proches (cosinus ≈ 1) et deux animaux différents produisent des vecteurs éloignés.

### 8.2 Architecture de l'embedding composite (464 dimensions)

Plutôt qu'utiliser uniquement DINOv2, le système construit un embedding **composite de 464 dimensions** combinant trois signaux complémentaires :

| Composante | Dimensions | Poids | Capture |
|---|---|---|---|
| **DINOv2-small** | 384 | **0.70** | Apparence sémantique globale (forme, posture) |
| **Histogramme HSV** | 48 | 0.20 | Distribution de couleur de robe |
| **LBP (Local Binary Pattern)** | 32 | 0.10 | Texture fine, robuste à l'éclairage |
| **TOTAL** | **464** | normalisé à 1 | |

**Justification des poids** : DINOv2 capture la majorité de l'information discriminative (0.70). HSV ajoute la couleur de robe (crucial pour distinguer une Charolaise blanche d'une Angus noire). LBP ajoute la texture (motif de robe).

**Conseil de tuning documenté** : pour des races à robe unie (Angus noir, Charolaise blanche), où la couleur seule discrimine fortement, il est recommandé d'ajuster `dino_weight=0.3, hsv_weight=0.5`.

### 8.3 DINOv2-small

- **Modèle** : `facebook/dinov2-small` (HuggingFace Transformers)
- **Dimensions** : 384
- **Pré-entraînement** : self-supervised, 142M images
- **Pooling** : moyenne sur la dimension des tokens (`last_hidden_state.mean(dim=1)`)
- **Normalisation** : L2 avec epsilon `1e-8`

### 8.4 Histogramme HSV (48-dim)

```python
def _hsv_hist(self, crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = []
    for ch in range(3):  # H, S, V séparément
        h = cv2.calcHist([hsv], [ch], None, [16], 
                         [0, 180] if ch == 0 else [0, 256])
        hist.append(h.flatten())
    return np.concatenate(hist)  # 48-dim, normalisé par somme
```

**Note de bug historique** : une ancienne version utilisait un histogramme 3D joint (16×16×16 = 4096 dim), beaucoup trop lourd et peu discriminant. La version actuelle utilise 3 histogrammes 1D séparés, plus compacts et plus stables.

### 8.5 LBP (Local Binary Pattern, 32-dim)

Le LBP est un descripteur de texture classique :

1. Pour chaque pixel, comparer aux 8 voisins → code binaire 8-bit
2. Calculer l'histogramme sur une grille 4×4 avec 8 bins → 128 dim
3. **Réduire à 32 dim** en moyennant par groupes de 4

Le calcul est vectorisé via `np.roll` pour les 8 décalages, et l'image est pré-floutée avec `cv2.GaussianBlur((3,3), 0)`.

### 8.6 Similarité pondérée par composante

Plutôt qu'un simple dot product sur les 464 dim concaténées, le système calcule la similarité **par composante** puis combine :

```python
def compare(self, embedding_a, embedding_b):
    # Découper en 3 slots
    a_dino, b_dino = embedding_a[:384],    embedding_b[:384]
    a_hsv,  b_hsv  = embedding_a[384:432], embedding_b[384:432]
    a_lbp,  b_lbp  = embedding_a[432:],    embedding_b[432:]
    
    # Cosinus par composante
    sim_dino = cosine(a_dino, b_dino)
    sim_hsv  = cosine(a_hsv,  b_hsv)
    sim_lbp  = cosine(a_lbp,  b_lbp)
    
    # Somme pondérée
    return (0.70 * sim_dino + 0.20 * sim_hsv + 0.10 * sim_lbp)
```

**Avantage** : égalise la contribution de chaque composante indépendamment de sa dimensionnalité. Sans cela, les 384 dim de DINOv2 domineraient massivement les 48 + 32 = 80 dim combinées HSV+LBP.

### 8.7 Seuils de décision

| Seuil | Valeur | Utilisation |
|---|---|---|
| `threshold` | **0.70** | Re-ID normal (haute confiance) |
| `loop_threshold` | 0.55 | Pendant 60 frames après rewind (permissif) |
| Match minimal | 0.55 | En dessous → nouveau bovin |

### 8.8 Mise à jour EMA (Exponential Moving Average)

L'embedding d'un bovin évolue dans le temps (angle de vue, éclairage). Pour garder une référence à jour sans dérive :

```python
def update(self, name, embedding, alpha=0.2):
    old = self.animals[name]["embedding"]
    new = (1.0 - alpha) * old + alpha * embedding
    new = new / (np.linalg.norm(new) + 1e-8)  # re-normaliser
```

- `alpha = 0.2` : 80% ancien + 20% nouveau → évolution lente
- **Cap à 30 mises à jour** par animal (`max_ema_updates=30`) pour éviter la dérive

### 8.9 Optimisations DINOv2

| Optimisation | Gain estimé | Condition |
|---|---|---|
| `torch.compile(mode="reduce-overhead")` | +20-30 % | CUDA, PyTorch ≥ 2.0 |
| FP16 (`model.half()`) | +20 % single / +10 % batch-8 | MPS |
| Batch processing (1 forward pour N crops) | dominant | toujours |
| `@torch.no_grad()` | évite le calcul de gradient | toujours |

---

## 9. Classification de race avec SigLIP-2

### 9.1 Le défi de la classification sans données d'entraînement

Le projet ne disposait **pas d'images annotées** de races bovines. Entraîner un classifieur classique était donc impossible. La solution retenue est le **zero-shot classification** : décrire chaque race en langage naturel et laisser le modèle VLM (Vision-Language Model) la reconnaître sans entraînement.

### 9.2 Évolution des approches

#### Tentative 1 : HSV heuristique (supprimée)
Classification par analyse de couleur de robe (noir, blanc, fauve...). Trop imprécise : Charolaise, Limousine et Salers ont toutes une robe fauve — indissociables par couleur seule.

#### Tentative 2 : CLIP ViT-B/32
Première approche zero-shot viable. Cependant, les scores sigmoid de CLIP sont **plats** (toutes races ≈ 0.5), rendant la marge de décision toujours inférieure au seuil → tout classé "Indéterminée".

#### Tentative 3 : SigLIP-2 So400m + Softmax (version finale)

- **Modèle** : `google/siglip2-so400m-patch16-384` (1136M paramètres)
- **Performance** : 84,1 % top-1 ImageNet zero-shot
- **Scoring** : softmax(température=0.01) sur similarités cosinus

### 9.3 Les 10 races supportées

Chaque race est décrite par un **prompt riche** en anglais encodant les discriminants visuels :

| Race | Robe | Origine | Usage |
|---|---|---|---|
| Holstein | Pie noire-blanc | Pays-Bas | Laitière (n°1 mondiale) |
| Charolaise | Blanche crème | Bourgogne | Bouchère |
| Limousine | Fauve | Limousin | Bouchère |
| Salers | Fauve rouge | Auvergne | Mixte |
| Angus | Noire | Écosse | Bouchère |
| Normande | Pie brune | Normandie | Mixte |
| Blonde d'Aquitaine | Froment | Aquitaine | Bouchère |
| Montbéliarde | Pie rouge | Franche-Comté | Laitière |
| Hereford | Rouge blanc-face | Angleterre | Bouchère |
| Aubrac | Fauve | Aubrac | Mixte |

Exemple de prompt (Holstein) :
> "A dairy cow with distinctive black and white patched coat pattern, large robust frame, typically black hooves, with clearly defined irregular black and white areas covering the body, the most recognizable dairy breed worldwide."

Ces prompts encodent la **couleur précise**, le **pattern** (pie/unie/bringée), la **morphologie**, et l'**usage** — autant de discriminants que SigLIP-2 peut évaluer.

### 9.4 Algorithme de classification

```python
def classify(self, crop_bgr):
    # 1. Embedder l'image
    emb = self._embed_image(crop_bgr)  # L2-normalisé
    
    # 2. Similarités cosinus avec les prompts pré-calculés
    sims = emb.cpu() @ self.text_emb.cpu().T  # (10,)
    
    # 3. Softmax avec température faible
    probs = softmax(sims / 0.01)  # τ=0.01 → distribution piquée
    
    # 4. Top-1 et marge
    top1_idx = argmax(probs)
    margin = probs[top1] - probs[top2]
    
    return {"race": BREEDS[top1_idx], "confidence": probs[top1], "margin": margin}
```

**Pourquoi softmax plutôt que sigmoid ?**

- **Sigmoid** : traite chaque classe indépendamment → scores ~0.5 pour toutes, pas de contraste
- **Softmax** : compétition entre classes → une race domine clairement

Avec τ=0.01 (très faible), le softmax produit une distribution fortement **piquée** : la vraie race obtient typiquement 58-80 %, la deuxième 10-20 %, soit une marge > 0.40.

### 9.5 Le seuil d'honnêteté

```python
CONFIDENCE_MARGIN_THRESHOLD = 0.15
```

Si `margin < 0.15`, le système **refuse de deviner** et retourne `"Indéterminée"`.

**Philosophie** : il vaut mieux dire "je ne sais pas" que donner une mauvaise réponse. Les races fauve (Charolaise, Limousine, Salers, Blonde d'Aquitaine) sont visuellement très proches. SigLIP-2, même performant, peut hésiter entre elles.

**Exemple de cas problématique** : sans le seuil, un bovin Salers serait classé Limousine avec 51 % vs 49 % — faux et trompeur. Avec le seuil, il est "Indéterminée", et l'utilisateur sait que la classification est incertaine.

### 9.6 Pré-calcul des embeddings texte

Les prompts des 10 races sont **encodés une seule fois** au démarrage :

```python
def _cache_text_embeddings(self):
    self.text_emb = []  # liste de vecteurs L2-normalisés
    for breed_name, info in BREEDS.items():
        emb = self._embed_text(info["prompt"])
        self.text_emb.append(emb)
    self.text_emb = torch.stack(self.text_emb)  # (10, D)
```

Ainsi, dans la boucle vidéo, seule l'embedding **image** est calculé (le produit matriciel `emb @ text_emb.T` est quasi instantané).

### 9.7 Chaîne de fallback

```python
_BreedEngine._load():
    try:
        # 1. SigLIP-2 So400m (1136M params)
        from transformers import AutoModel, AutoProcessor
        self.model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-384")
    except:
        try:
            # 2. CLIP ViT-B/32 (fallback)
            self.model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        except:
            # 3. HSV heuristique (dernier recours)
            return None
```

Le système dégrade gracieusement : SigLIP-2 → CLIP → HSV → aucun.

### 9.8 Performance

- **Latence** : ~22 ms par classification (MPS, M1 Pro)
- **Fréquence** : uniquement sur **nouveau bovin** (première détection), pas par frame
- **Impact sur FPS** : négligeable (1 classification toutes les N frames où N = frames entre nouveaux bovins)

---

## 10. Génération de noms et persistance d'identité

### 10.1 Le problème de l'identité stable

Un bovin détecté dans la vidéo A doit garder le même nom quand il réapparaît dans la vidéo B, même après :

- Redémarrage de l'application
- Réinitialisation de la base de données
- Changement de vidéo source
- Rebouclage de la vidéo

### 10.2 Architecture en 3 couches

#### Couche 1 : Re-ID embedding (identification visuelle)
Voir section 8. Produit une empreinte visuelle stable.

#### Couche 2 : Base de données par vidéo (`.pkl`)
Chaque vidéo a sa propre base de données :

```python
def db_path_for_source(source) -> str:
    # Webcam 0 → "cattle_db_webcam0.pkl"
    # Fichier "107414-678258609_medium.mp4" → "cattle_db_107414-678258609_medium.pkl"
    # Nom > 50 chars → hash MD5 → "cattle_db_video_<hash>.pkl"
```

Quand on change de vidéo, la DB actuelle est **sauvegardée** puis une nouvelle (ou existante) est **chargée** pour la nouvelle source — sans perte de données.

#### Couche 3 : Compteur global (`names_counter.json`)

```json
{"counter": 43}
```

Un simple compteur persistant qui ne fait **que croître**. Garantit que `Boeuf_044` ne sera jamais attribué si `Boeuf_043` a déjà existé, même après reset de DB.

```python
class GlobalCounter:
    def next(self) -> int:
        self.value += 1
        self.save()  # persist immédiatement
        return self.value
```

### 10.3 La pool de noms (~120 noms)

Le système attribue des **noms propres lisibles** plutôt que des IDs techniques. La pool contient ~120 noms équilibrés entre cultures européennes et africaines :

- **Européens (50)** : Marguerite, Aurelius, Charline, Daphne, Bastille, Beethoven, Ulysse, Sybille...
- **Africains (50)** : noms Wolof, Bambara, Mandinka, Swahili, Yoruba
- **Extras (20)** : noms de minéraux/phénomènes (overflow)

**Aucun emoji** — compatibilité terminal/UNIX.

### 10.4 Mapping clé → nom

La classe `NameGenerator` mappe chaque clé `Boeuf_NNN` vers un nom de la pool :

```python
class NameGenerator:
    def __init__(self, animals: dict):
        self._mapping = {}
        # Trier les animaux par ordre de création (first_seen)
        sorted_keys = sorted(animals.keys(), key=lambda k: animals[k]["first_seen"])
        for i, key in enumerate(sorted_keys):
            # Mapping par NUMÉRO de clé, pas par index trié
            key_num = int(key.split("_")[1])  # "Boeuf_004" → 4
            self._mapping[key] = NAME_POOL[key_num - 1]  # 0-indexed
```

**Subtilité importante** : le mapping se fait par **numéro de clé** (`Boeuf_004 → NAME_POOL[3]`), pas par index de tri. Ainsi, après un changement de vidéo qui vide la DB, les nouveaux `Boeuf_005+` obtiennent des noms *différents* des `Boeuf_001-004` de la vidéo précédente.

### 10.5 Séquence complète d'identification d'un nouveau bovin

```
1. DINOv2 calcule l'embedding (464-dim)
2. db.match(embedding, threshold=0.70) → (None, 0.45) → pas de match
3. next_bovin_key() → "Boeuf_044"  (compteur global)
4. classify_breed(crop) → {"race": "Holstein", "confidence": 0.72}
5. db.add("Boeuf_044", embedding, breed="Holstein", ...)
6. db.save()  → persiste dans cattle_db_<video>.pkl
7. name_gen.get("Boeuf_044") → "Aurelius"
8. Émission event NEW : {"type": "NEW", "name": "Aurelius", "breed": "Holstein"}
```

---

## 11. Pipeline de traitement vidéo

### 11.1 Vue d'ensemble du `detection_loop`

Le cœur du système est la fonction `detection_loop(args)` dans `processor.py` (~600 lignes). C'est une boucle infinie qui :

1. Capture une frame
2. Détecte les bovins
3. Calcule les embeddings Re-ID (asynchrones)
4. Match/ ajoute à la DB
5. Classifie les comportements
6. Annote la frame
7. Encode en JPEG
8. Met à jour l'état global

### 11.2 Initialisation

```python
def detection_loop(args):
    # 1. Résoudre le device (MLX auto-detect)
    use_mlx = _mlx_available() and (args.device in ("auto", "mlx"))
    
    # 2. Construire le détecteur
    if use_mlx:
        detector = CattleDetectorMLX(args.yolo_model)
    else:
        detector = CattleDetector(args.yolo_model)
    
    # 3. Construire le moteur Re-ID
    reid = CattleReID(args.dino_model)
    
    # 4. Charger la DB par vidéo
    initial_db_path = db_path_for_source(args.source)
    db = EmbeddingDatabase(initial_db_path, reid_engine=reid)
    
    # 5. Synchroniser le compteur global avec la DB
    counter.sync_with_db(db.animals)  # évite les collisions
    
    # 6. Valider les dimensions d'embedding
    db.validate_dim(reid.TOTAL_DIM)  # purge si mismatch (changement de modèle)
    
    # 7. Lancer le worker Re-ID asynchrone
    reid_worker = ReIDWorker(reid)
    
    # 8. Précharger SigLIP-2
    get_clip_engine()
```

### 11.3 Boucle principale (par frame)

```python
while True:  # avec auto-recovery extérieur
    try:
        # --- Hot-reload des settings ---
        apply_desired_settings()
        if source_changed: switch_source()
        if device_changed: switch_device()
        
        # --- Lecture frame ---
        ret, frame = cap.read()
        if not ret:
            _recover_capture()  # webcam: reconnecter; fichier: rewind
            continue
        
        # --- Détection de rewind ---
        current_pos = cap.get(CAP_PROP_POS_FRAMES)
        if current_pos < last_pos:
            loop_detected_at_frame = frame_count
            track_id_to_name.clear()
        
        # --- Skip adaptatif ---
        effective_skip = min(_current_skip, _max_skip)
        if frame_count % (effective_skip + 1) != 0:
            continue
        
        # --- Détection YOLO ---
        result = detector.detect(frame, conf=args.conf, imgsz=args.imgsz)
        
        # Extraire boxes, ids, masks (style torch, via shims si MLX)
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().numpy()
        masks = result.masks.data.cpu().numpy()
        
        # --- Phase 1 : Collecte des crops ---
        crops_to_submit = {}
        for det_idx, tid in enumerate(track_ids):
            crop = _crop_from_box(frame, boxes[det_idx])
            if tid not in track_id_to_name:
                # Nouveau track → Re-ID
                crops_to_submit[tid] = crop
            elif frame_count % args.embed_every == 0:
                # Re-embed périodique → mise à jour EMA
                crops_to_submit[tid] = crop
        
        # --- Phase 2 : Re-ID asynchrone ---
        if reid_worker.has_failed():
            embeddings = reid.get_embedding_batch(list(crops_to_submit.values()))
        else:
            reid_worker.submit_batch(crops_to_submit)
            embeddings = reid_worker.collect_ready()
        
        # --- Phase 3 : Match / EMA / Annotate ---
        for tid, emb in embeddings.items():
            if tid not in track_id_to_name:
                # Match contre la DB
                name, sim = db.match(emb, threshold=eff_threshold)
                if name is None:
                    # Nouveau bovin
                    key = next_bovin_key()
                    breed = classify_breed(crop)
                    db.add(key, emb, breed=breed["race"], ...)
                    db.save()
                    name = name_gen.get(key)
                    emit_event("NEW", name, breed)
                else:
                    emit_event("MATCH", name)
                track_id_to_name[tid] = name
            else:
                # Mise à jour EMA
                if updates_count[tid] < MAX_UPDATES:
                    db.update(track_id_to_name[tid], emb, alpha=0.2)
        
        # --- Annotation ---
        annotate_frame(frame, boxes, track_ids, masks, track_id_to_name)
        
        # --- Comportement ---
        behaviors = analyze_behavior(boxes, track_ids, frame.shape)
        
        # --- Encode JPEG off-thread ---
        future = jpeg_pool.submit(cv2.imencode, ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        # ... collect previous future non-bloquant ...
        
        # --- State update ---
        with STATE["frame_lock"]:
            STATE["frame_jpg"] = jpeg_bytes
        
        # --- FPS ---
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)
        
        # --- Skip tuning ---
        if fps_smooth < 20: _current_skip = min(_current_skip + 1, 4)
        elif fps_smooth > 28: _current_skip = max(_current_skip - 1, 0)
    
    except Exception as e:
        emit_event("CRASH", traceback.format_exc())
        time.sleep(2)
        cap.release()
        cap = open_capture(source)
```

### 11.4 Le ReIDWorker asynchrone

Un thread daemon séparé découple DINOv2 (MPS) de la boucle vidéo (YOLO MLX) :

```python
class ReIDWorker:
    def __init__(self, reid_engine):
        self.reid = reid_engine
        self.queue = Queue(maxsize=64)  # backlog limité
        self.results = {}  # {track_id: embedding}
        self.thread = Thread(target=self._loop, daemon=True, name="reid-worker")
        self.thread.start()
    
    def _loop(self):
        while not self._stop:
            batch = self.queue.get()  # bloquant
            try:
                embeddings = self.reid.get_embedding_batch(batch.crops)
                self.results.update(zip(batch.track_ids, embeddings))
            except:
                self._failed = True
    
    def submit_batch(self, crops_by_tid):
        try:
            self.queue.put_nowait(Batch(crops_by_tid))  # non-bloquant
        except Full:
            pass  # drop silencieux, on réessaiera
    
    def collect_ready(self):
        # Récupère non-bloquant les embeddings prêts
        ready = {}
        for tid in list(self.results):
            if tid in self.results:
                ready[tid] = self.results.pop(tid)
        return ready
```

**Justification** : DINOv2 tourne sur MPS (PyTorch) tandis que YOLOv26 tourne sur MLX (Metal). Les deux backends GPU peuvent fonctionner en parallèle — le worker exploite cette concordance.

---

## 12. Classification des comportements

### 12.1 Les 7 comportements reconnus

| Code | Label | Description |
|---|---|---|
| 0 | **couché** | Allongé, immobile longuement |
| 1 | **pâture** | Broute, tête basse |
| 2 | **boit** | À proximité d'un point d'eau |
| 3 | **immobile** | Debout sans bouger |
| 4 | **marche** | Déplacement lent |
| 5 | **court** | Déplacement rapide |
| 6 | **rué** | Sprint, panique |

### 12.2 Features cinématiques

La classification utilise 3 features calculées par track :

1. **`speed`** (px/s) : vitesse de déplacement du centroïde
2. **`aspect`** (ratio) : `box_width / box_height` (un bovin couché a un aspect > 1.7)
3. **`rel_y`** : position verticale relative (`center_y / frame_height`, 0 = haut, 1 = bas)

### 12.3 Arbre de décision Numba JIT

```python
@_njit(cache=True)
def _classify_behavior(speed, aspect, rel_y, immobile_dur):
    if aspect > 1.7 and speed < 5.0 and immobile_dur > 3.0:
        return 0  # couché
    if speed < 6.0 and aspect > 1.4:
        return 1  # pâture
    if speed < 4.0 and aspect > 1.3 and rel_y > 0.6:
        return 2  # boit (proximité sol = point d'eau)
    if speed < 5.0:
        return 3  # immobile
    if speed < 25.0:
        return 4  # marche
    if speed < 80.0:
        return 5  # court
    return 6  # rué
```

**Justification Numba** : cette fonction est appelée pour chaque track à chaque frame. Sans JIT, l'interpréteur Python serait 5-10× plus lent.

### 12.4 Lissage par vote majoritaire

Les comportements instantanés sont bruités. Le système maintient un **historique roulant de 5 frames** par track et applique un vote majoritaire :

```python
history[tid].append(instant_code)
if len(history[tid]) > 5:
    history[tid].pop(0)
smoothed = mode(history[tid])  # vote majoritaire
```

### 12.5 Détection d'immobilité

Un bovin est considéré "immobile" si son déplacement cumulé sur les dernières frames est < 30 px :

```python
displacement = _total_displacement(track_history[tid])  # JIT
immobile_dur = time_since(displacement > 30)
```

---

## 13. Analyse de données et tableau de bord

### 13.1 Le module analytics

Le module `analytics.py` collecte en arrière-plan des données pour alimenter le tableau de bord :

```python
class AnalyticsCollector:
    def __init__(self):
        self.state = AnalyticsState()
        self.thread = Thread(target=self._sample_loop, daemon=True)
    
    def _sample_loop(self):
        while True:
            time.sleep(2.0)  # sample_interval
            self.sample_fps(STATE["fps"])
            for animal in STATE["active_animals"]:
                self.sample_detection(animal)
            # ...
```

### 13.2 Données collectées

| Type | Détail | Persistance |
|---|---|---|
| **FPS history** | Toutes les 2s, 600 points max | `web/data/history.json` |
| **Race counts** | Comptage par race détectée | history.json |
| **Activity counts** | Distribution des comportements | history.json |
| **Timeline events** | NEW/MATCH/CRASH/INFO, 200 max | history.json |
| **Spatial samples** | Position (x, y) normalisée + coat_type | en mémoire |

### 13.3 Endpoints API analytics

| Route | Données retournées |
|---|---|
| `/api/dashboard` | KPIs, fps_history, race_counts, activity_counts, timeline, breed_colors |
| `/api/heatmap` | Grille 20×20 avec compte par cellule + coat_type |
| `/api/profiles` | Profils individuels : breed, count, videos, activities_pct |

### 13.4 Persistance cross-session

`web/data/history.json` (26 Ko sur le projet de test) contient un instantané qui survit aux redémarrages. Il inclut :

- ~4300 détections cumulées
- Distribution des races : Fauve uni, Indéterminée, Bringée, Angus, Normande
- Distribution des activités : marche, immobile, court, rué, pâture, couché
- Timeline riche avec événements (noms : Aurelius, Charline, Daphne, Bastille, Beethoven...)

---

## 14. Carte de chaleur spatiale (heatmap)

### 14.1 Principe

La heatmap représente la **densité spatiale** des bovins dans le champ de la caméra. Elle permet d'identifier :

- Les zones de repos (concentration élevée)
- Les zones de passage (concentration moyenne)
- Les zones évitées (concentration nulle)

### 14.2 Grille 20×20

Le système discrétise l'image en une grille de **20×20 = 400 cellules**. Chaque détection incrémente le compteur de sa cellule :

```python
grid_x = int(center_x / frame_width * 20)
grid_y = int(center_y / frame_height * 20)
heatmap[grid_y][grid_x] += 1
```

### 14.3 Rendu canvas côté frontend

Le frontend dessine la heatmap avec un effet **kernel-density** (gaussian radial gradients) :

```javascript
function drawHeatmap(canvas, cells) {
    const ctx = canvas.getContext('2d');
    ctx.globalCompositeOperation = 'lighter';  // blending additif
    
    for (const cell of cells) {
        const x = cell.x / 20 * width;
        const y = cell.y / 20 * height;
        const intensity = cell.count / maxCount;
        const radius = 30 + intensity * 40;
        
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
        gradient.addColorStop(0, heatColor(intensity));  // rouge si intense
        gradient.addColorStop(1, 'rgba(0,0,0,0)');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(x - radius, y - radius, radius * 2, radius * 2);
    }
}

function heatColor(t) {
    // Palette 3 stops : vert → ambre → rouge brique
    if (t < 0.5) return interpolate('#16a34a', '#f59e0b', t * 2);
    return interpolate('#f59e0b', '#b91c1c', (t - 0.5) * 2);
}
```

### 14.4 Annotations spatiales

La heatmap est annotée avec :

- **Axes** : Haut/Milieu/Bas (vertical), Gauche/Centre/Droite (horizontal)
- **Légende** : barre de gradient vert→rouge
- **Retina-aware** : scaling DPR (Device Pixel Ratio) pour netteté sur écrans HiDPI
- **État vide** : placeholder si aucune donnée

---

## 15. Serveur Flask et API REST

### 15.1 Architecture du serveur

Le serveur Flask (`app.py`, 758 lignes) est le pivot central :

```
app.py
├─ Thread principal : serveur Flask (port 8100)
├─ Thread daemon : detection_loop (pipeline vidéo)
├─ JSON provider : NumpyJSONProvider (sérialise numpy)
└─ Static : sert web/public/ (HTML/JS/CSS)
```

### 15.2 Routes API complètes

| Route | Méthode | Description |
|---|---|---|
| `/` | GET | Sert `index.html` |
| `/<path:filename>` | GET | Sert les assets statiques |
| `/video_feed` | GET | Snapshot JPEG unique (polling) |
| `/api/stats` | GET | État temps-réel : fps, source, animaux, événements |
| `/api/devices` | GET | Liste des devices (mlx, cuda, mps, cpu) |
| `/api/device` | POST | Change de device à l'exécution |
| `/api/upload-video` | POST | Upload d'une vidéo (multipart) |
| `/api/source/webcam` | POST | Bascule vers webcam |
| `/api/videos` | GET | Liste des vidéos disponibles |
| `/api/source/file` | POST | Bascule vers un fichier vidéo |
| `/api/diag` | GET | Dump diagnostique de l'état |
| `/api/db/reset` | POST | Purge la DB de la source courante |
| `/api/rematch` | POST | Force le re-Re-ID de tous les tracks |
| `/api/restart` | POST | Redémarre le serveur (via watcher) |
| `/api/settings` | GET | Settings actuels + désirés + disponibles |
| `/api/settings` | POST | Hot-reload des paramètres |
| `/api/breeds` | GET | Liste des 10 races |
| `/api/breeds/<name>` | GET | Détail d'une race |
| `/api/animals` | GET | Tous les animaux en DB |
| `/api/dashboard` | GET | Données du tableau de bord |
| `/api/heatmap` | GET | Grille spatiale 20×20 |
| `/api/profiles` | GET | Profils individuels par bovin |
| `/api/bench` | GET | Benchmark YOLO11 vs YOLO26 vs DINOv2 |

### 15.3 Le pattern desired/current pour le hot-reload

Les settings suivent un pattern à **deux états** :

- **`desired_*`** : ce que l'utilisateur veut (modifié via POST `/api/settings`)
- **`*_current`** : ce qui est effectivement appliqué

```python
# POST /api/settings
STATE["desired_imgsz"] = 960  # désiré

# Dans detection_loop
def apply_desired_settings():
    if STATE["desired_imgsz"] != STATE["imgsz_current"]:
        STATE["imgsz"] = STATE["desired_imgsz"]  # applique
        STATE["imgsz_current"] = STATE["desired_imgsz"]  # confirme
```

Le frontend affiche un badge "pending" tant que `desired != current`.

### 15.4 Le endpoint `/video_feed` — JPEG polling

Plutôt qu'un flux MJPEG (multipart), le système retourne un **JPEG unique par requête** :

```python
@app.route("/video_feed")
def video_feed():
    frame = STATE.get("frame_jpg")
    if frame is None:
        # Placeholder 1×1 transparent (évite HTTP 204 mal géré par certains navigateurs)
        return Response(JPEG_1X1_PLACEHOLDER, mimetype="image/jpeg")
    
    return Response(frame, mimetype="image/jpeg",
                   headers={"Cache-Control": "no-store"})
```

**Raison** : Cloudflare et certains proxies tuent les connexions keep-alive > 100s (erreur 524). Le polling JPEG évite ce problème.

### 15.5 Sérialisation numpy

```python
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)
```

Permet de retourner directement des arrays numpy dans les réponses JSON Flask.

---

## 16. Interface utilisateur web

### 16.1 Stack frontend

- **HTML5** sémantique (`index.html`, 13 Ko)
- **CSS3** avec variables custom, dark/light theme (`styles.css`, 18 Ko)
- **JavaScript vanilla** (pas de framework) (`app.js`, 33 Ko)
- **Chart.js 4.4** (CDN) pour les graphiques
- **Canvas natif** pour la heatmap

### 16.2 Layout principal

```
┌─────────────────────────────────────────────────────────────┐
│ TOPBAR : Logo BT | Source | Device | FPS | Frames | [theme] │
├─────────────────────────────────────────────────────────────┤
│ SOURCE BAR : [Video ▼] [Upload] [Webcam] | [Device ▼] [Reset]│
├──────────────────────────────────────┬──────────────────────┤
│                                       │  PARAMÈTRES (live)  │
│                                       │  ├ YOLO model       │
│        FLUX VIDÉO ANNOTÉ              │  ├ imgsz            │
│        (img pollant /video_feed)      │  ├ embed_every      │
│                                       │  ├ threshold        │
│        ┌──────────────────┐           │  └ conf             │
│        │ Marguerite       │           ├──────────────────────┤
│        │ Holstein  72%    │           │  ANIMAUX DÉTECTÉS   │
│        └──────────────────┘           │  ├ Marguerite H. 72%│
│                                       │  ├ Aurelius C. 65%  │
│                                       │  └ ...              │
│                                       ├──────────────────────┤
│                                       │  JOURNAL            │
│                                       │  [NEW] Marguerite   │
│                                       │  [MATCH] Aurelius   │
│                                       ├──────────────────────┤
│                                       │  ACTIVITÉS         │
│                                       │  Marguerite: pâture│
│                                       │  Aurelius: marche  │
└──────────────────────────────────────┴──────────────────────┘
```

### 16.3 Panneau Dashboard (slide-in)

Bouton "Analytics" → panneau latéral (50vw) avec :

- **4 KPIs** : bovins uniques / bovins visibles / FPS moyen / uptime
- **Graphique FPS** (Chart.js line, 10 min d'historique)
- **Donut des races** (avec légende custom)
- **Barres des activités** (triées desc)
- **Heatmap spatiale** (canvas natif, kernel-density)
- **Timeline** filtrable (ALL/NEW/MATCH/CRASH/INFO)

### 16.4 Panneau Statistiques (slide-in)

Profils individuels par bovin :

- Nom + race
- Compte de détections
- Clé interne (`Boeuf_042`)
- Nombre de vidéos où apparaît
- **Barres d'activités** en pourcentage

### 16.5 Polling cadencé

| Fonction | Fréquence | Endpoint |
|---|---|---|
| `refreshStats()` | 1000 ms | `/api/stats` |
| `refreshDashboard()` | 3000 ms (si ouvert) | `/api/dashboard` + `/api/heatmap` |
| `refreshProfiles()` | 5000 ms (si ouvert) | `/api/profiles` |
| Stream vidéo | ~40 ms (25 FPS) | `/video_feed` |

### 16.6 Thème dark/light

```javascript
// Anti-flash : applique avant le render
const theme = localStorage.getItem('theme') || 'dark';
document.documentElement.setAttribute('data-theme', theme);
```

Variables CSS custom pour tous les couleurs :

```css
[data-theme="dark"]  { --bg: #0f1115; --accent: #16a34a; ... }
[data-theme="light"] { --bg: #ffffff; --accent: #15803d; ... }
```

### 16.7 Hot-settings côté UI

```javascript
async function pushSetting(key, value) {
    if (pushInFlight) return;  // mutex
    pushInFlight = true;
    try {
        await fetch('/api/settings', {
            method: 'POST',
            body: JSON.stringify({ [key]: value })
        });
    } finally {
        pushInFlight = false;
    }
}
```

Le mutex `pushInFlight` sérialise les push pour éviter les races conditions.

---

## 17. Application desktop Tauri

### 17.1 Pourquoi Tauri ?

| Critère | Electron | Tauri |
|---|---|---|
| Taille binaire | ~150 Mo | **~1.8 Mo** |
| RAM | ~200 Mo | ~50 Mo |
| Backend | Node.js (bundled) | Rust (système) |
| WebView | Bundled Chromium | Native (WKWebView/WebView2) |
| Démarrage | ~2s | ~0.3s |

Tauri est **80× plus léger** qu'Electron — crucial pour une app qui lance déjà un worker Python gourmand.

### 17.2 Architecture Tauri du projet

```rust
// src-tauri/src/main.rs (~266 lignes)
const WORKER_PORT: u16 = 8100;
const HEALTH_TIMEOUT_SECS: u64 = 60;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(PythonWorker(Mutex::new(None)))
        .setup(|app| {
            let window = app.get_webview_window("main").unwrap();
            
            // 1. Afficher le splash
            window.navigate("http://tauri.localhost/splash.html")?;
            
            // 2. Lancer le worker Python
            let child = spawn_worker(app)?;
            app.state::<PythonWorker>().0.lock().unwrap().replace(child);
            
            // 3. Attendre qu'il soit prêt
            let app_handle = app.handle().clone();
            thread::spawn(move || {
                if wait_for_worker(HEALTH_TIMEOUT_SECS) {
                    // 4. Naviguer vers l'UI
                    app_handle.get_webview_window("main")
                        .unwrap()
                        .navigate(WORKER_URL)?;
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::Destroyed = event {
                // 5. Tuer le worker à la fermeture
                let worker = window.app_handle().state::<PythonWorker>();
                if let Some(child) = worker.0.lock().unwrap().take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### 17.3 Détection cross-platform du Python

```rust
fn find_python() -> Option<PathBuf> {
    // 1. Venv en priorité
    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        let python = format!("{}/bin/python", venv);
        if Path::new(&python).exists() { return Some(python.into()); }
        let python = format!("{}/Scripts/python.exe", venv);  // Windows
        if Path::new(&python).exists() { return Some(python.into()); }
    }
    // 2. .venv local
    for candidate in ["./.venv/bin/python", "./.venv/Scripts/python.exe"] {
        if Path::new(candidate).exists() { return Some(candidate.into()); }
    }
    // 3. Système
    which("python3").or_else(|| which("python"))
}
```

### 17.4 Health check en 2 phases

```rust
fn wait_for_worker(timeout: u64) -> bool {
    let start = Instant::now();
    
    // Phase 1 : port TCP ouvert
    while start.elapsed().as_secs() < timeout {
        if TcpStream::connect(("127.0.0.1", WORKER_PORT)).is_ok() {
            break;
        }
        thread::sleep(Duration::from_millis(500));
    }
    
    // Phase 2 : fps > 0 (modèles chargés, boucle démarrée)
    while start.elapsed().as_secs() < timeout {
        if let Ok(output) = Command::new("curl")
            .args(["-s", &format!("{}/api/stats", WORKER_URL)])
            .output() {
            let json = String::from_utf8_lossy(&output.stdout);
            if let Some(fps) = parse_fps(&json) {
                if fps > 0.0 { return true; }  // VRAIMENT prêt
            }
        }
        thread::sleep(Duration::from_secs(1));
    }
    false
}
```

**Pourquoi 2 phases ?** Le port TCP s'ouvre dès que Flask démarre (~1s), mais les modèles MLX/SigLIP-2 prennent 15-30s à charger. Sans la phase 2, l'UI afficherait `fps: 0` pendant ce temps — mauvaise UX.

### 17.5 Splash screen

Pendant le démarrage du worker, Tauri affiche `splash.html` :

- Logo "BT" pulsant (animation CSS)
- Messages rotatifs toutes les 2.5s :
  - "Chargement des modèles MLX..."
  - "Initialisation YOLO26 (Metal GPU)..."
  - "Chargement DINOv2 (Re-ID)..."
  - "Chargement CLIP (reconnaissance de races)..."
  - "Démarrage de la boucle de détection..."

### 17.6 Configuration bundle

```jsonc
// tauri.conf.json
{
  "productName": "Boeuf Tracker",
  "version": "2.0.0",
  "identifier": "com.boeuf.tracker",
  "build": {
    "frontendDist": "../web/public"
  },
  "app": {
    "windows": [{
      "title": "Boeuf Tracker",
      "width": 1280, "height": 800,
      "minWidth": 900, "minHeight": 600
    }]
  },
  "bundle": {
    "targets": "all",  // macOS .app/.dmg + Win .exe/.msi + Linux .AppImage/.deb
    "icon": ["icons/icon.icns", "icons/icon.ico", "icons/icon.png"]
  }
}
```

### 17.7 Limitation assumée : Python requis côté utilisateur

Le bundle Tauri **n'embarque pas** Python ni les modèles. Il suppose que l'utilisateur a :

- Python 3.10+ installé
- Un `.venv` configuré avec les dépendances
- Les poids des modèles (YOLO, DINOv2, SigLIP-2)

C'est un compromis assumé : packager PyInstaller une stack ML aussi lourde (torch, transformers, mlx...) génère un exécutable de plusieurs Go, instable. Le wrapper dev est plus pragmatique.

---

## 18. Optimisations de performance

### 18.1 Tableau récapitulatif

| Optimisation | Localisation | Gain estimé |
|---|---|---|
| MLX/Metal pour YOLOv26 | detector.py | **2.6× vs PyTorch/MPS** |
| DINOv2 batch processing | reid.py | dominant (N forwards → 1) |
| ReIDWorker asynchrone | reid_worker.py | découple Re-ID du loop |
| `torch.compile` (CUDA) | reid.py | +20-30 % |
| FP16 sur MPS | reid.py | +20 % |
| Numba JIT comportements | processor.py | ×5-10 vs Python pur |
| Annotation ROI-scoped | processor.py | **29 ms → 9.5 ms/frame** |
| BLAS vectorisé match() | database.py | O(N) → 1 BLAS |
| Skip adaptatif | processor.py | 25 FPS garanti |
| JPEG encode off-thread | processor.py | -3 ms/frame |
| EMA cap 30 updates | processor.py | borne la dérive |
| Pré-calcul embeddings texte | breed.py | 0 ms in-loop |
| DB par vidéo (lazy load) | processor.py | pas de re-Re-ID au switch |

### 18.2 Détail : annotation ROI-scoped

L'annotation naïve dessine le mask sur toute l'image :

```python
# NAÏF (~29 ms) :
mask_full = cv2.resize(mask, (W, H))
overlay = frame.copy()
overlay[mask_full > 0] = color
cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
```

L'optimisation restreint l'opération à la **bounding box** uniquement :

```python
# OPTIMISÉ (~9.5 ms) :
x1, y1, x2, y2 = box
roi = frame[y1:y2, x1:x2]
mask_roi = cv2.resize(mask, (x2-x1, y2-y1))
overlay_roi = roi.copy()
overlay_roi[mask_roi > 0] = color
cv2.addWeighted(overlay_roi, 0.08, roi, 0.92, 0, roi)
```

Gain : **−65 %** sur l'annotation.

### 18.3 Détail : match() vectorisé

Version naïve (boucle Python) :

```python
# NAÏF : N dot products Python
best_name, best_sim = None, -1
for name, stored in self.animals.items():
    sim = cosine(query, stored.embedding)
    if sim > best_sim:
        best_sim, best_name = sim, name
```

Version vectorisée (1 BLAS) :

```python
# OPTIMISÉ : 1 opération matricielle
M = np.stack([d["embedding"] for d in animals.values()])  # (N, 464)
norms = np.linalg.norm(M, axis=1)  # (N,)
sims = (M @ query) / (norms * query_norm + 1e-8)  # 1 BLAS
best_idx = np.argmax(sims)
```

Pour N = 50 bovins, la vectorisation est **~10× plus rapide** (0.1 ms vs 1 ms). Pour N ≤ 10, la boucle reste compétitive (overhead numpy > gain BLAS), d'où un fallback `_match_loop()` pour petit N.

### 18.4 Détail : skip adaptatif

```python
_target_fps = 25
_current_skip = 0  # adjustable 0-4
_max_skip = args.skip_frames  # défaut 0

# À chaque frame :
effective_skip = min(_current_skip, _max_skip)
if frame_count % (effective_skip + 1) != 0:
    continue  # skip cette frame

# Tuning :
if fps_smooth < 20 for 3 frames:
    _current_skip = min(_current_skip + 1, 4)  # ralentir
elif fps_smooth > 28 for 5 frames:
    _current_skip = max(_current_skip - 1, 0)  # accélérer
```

**Objectif** : maintenir 25 FPS en sacrifiant des frames si nécessaire. Surcharge GPU → skip plus → moins de travail → FPS remonte.

---

## 19. Robustesse et gestion d'erreurs

### 19.1 Auto-récupération du pipeline

La `detection_loop` est enveloppée dans **deux boucles while** :

```python
while True:  # boucle extérieure = récupération
    try:
        while True:  # boucle intérieure = pipeline normal
            # ... traitement frame ...
    except Exception as e:
        emit_event("CRASH", traceback.format_exc().splitlines()[-1])
        time.sleep(2)
        cap.release()
        cap = open_capture(source)
        # continue la boucle extérieure → redémarre
```

Ainsi, un crash (frame corrompue, OOM, etc.) ne tue pas le système — il logge, attend 2s, et reprend.

### 19.2 Récupération de source vidéo

```python
def read_with_recovery(cap, src):
    ret, frame = cap.read()
    if not ret:
        if isinstance(src, int):  # webcam
            time.sleep(2)
            cap = open_capture(src)  # reconnect
            time.sleep(3)  # warmup webcam
        else:  # fichier
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind
        return None
    return frame
```

### 19.3 Récupération du worker Re-ID

Si `ReIDWorker` crash (OOM GPU par ex.) :

```python
if reid_worker.has_failed():
    # Fallback synchrone
    embeddings = reid.get_embedding_batch(crops)
else:
    reid_worker.submit_batch(crops)
    embeddings = reid_worker.collect_ready()
```

Le système dégrade gracieusement : async → sync, sans interruption.

### 19.4 Chaîne de fallback des modèles

**Détection** :
```
YOLOv26 MLX → YOLOv11 PyTorch (si MLX indisponible)
```

**Re-ID** :
```
DINOv2 + HSV + LBP → DINOv2 seul → HSV seul
```

**Race** :
```
SigLIP-2 So400m → CLIP ViT-B/32 → HSV heuristique → None
```

**Device** :
```
MLX (Metal) → CUDA → MPS → CPU
```

Chaque transition est **try/except guarded**, assurant que le système fonctionne même sur une machine minimale.

### 19.5 Validation des dimensions d'embedding

Si l'utilisateur change de modèle Re-ID (ex. : DINOv2-small → DINOv2-base), la dimension change (384 → 768). La DB détecte et purge :

```python
def validate_dim(self, expected_dim):
    bad = [n for n, d in self.animals.items() 
           if d["embedding"].shape[0] != expected_dim]
    for name in bad:
        del self.animals[name]
    self.save()
```

---

## 20. Évolution chronologique du projet

### 20.1 Timeline des 55 commits

| Date | Commits | Thématique |
|---|---|---|
| **24 juin 2026** | 4 | MVP initial (Flask + YOLO + DINOv2 + UI DaisyUI) |
| **26 juin 2026** | 3 | Settings live + notebook Colab |
| **27 juin 2026** | 18 | Bataille Cloudflare/ streaming (JPEG polling, Gradio, ngrok) |
| **13 juillet 2026** | 16 | Checkpoints intermédiaires (travail préparatoire) |
| **15 juillet 2026** | 13 | **Vague de features majeures** (perf, UI, noms, breeds, Tauri) |
| **16 juillet 2026** | 1 | Documentation finale + PPTX |

### 20.2 Phases de développement

#### Phase 0 — MVP (24 juin)
- Flask + YOLO11 + DINOv2 + cosine matching
- UI DaisyUI basique
- 3 vidéos de test

#### Phase 1 — Settings + Colab (26 juin)
- Hot-reload des paramètres
- Notebook Google Colab pour GPU distant

#### Phase 2 — Cloudflare Hardening (27 juin)
Problème : les flux MJPEG sont tués par Cloudflare (erreur 524).

Solutions tentées :
1. ✅ **JPEG polling** (remplace MJPEG) — solution retenue
2. Placeholder 1×1 (évite HTTP 204 mal géré)
3. Ajout Gradio comme alternative UI
4. Pin Gradio < 5.36 (bug manifest.json)
5. Switch gradio.live → ngrok

#### Phase 3 — Préparation (13 juillet)
16 checkpoints de travail intermédiaire (autosaves).

#### Phase 4 — Production Refactor (15 juillet) — le gros du travail

**Features majeures livrées en une journée :**

| Commit | Feature | Impact |
|---|---|---|
| `031cd0b` | ReIDWorker async + annotation ROI | **performance** |
| `d139341` | Architecture API + Bun gateway + UI dark/light | **architecture** |
| `7495aa4` | Générateur de noms (~100 noms) | **UX** |
| `3a972da` | Dashboard analytics + heatmap | **analytics** |
| `1e86254` | CLIP zero-shot breeds | **IA** |
| `47b4ddf` | **App desktop Tauri** | **déploiement** |
| `f145e2f` | Compteur global persistant | **persistance** |
| `7ed6865` | **SigLIP-2 So400m** (remplace CLIP) | **IA** |
| `82f0e72` | DB par vidéo + Re-ID resserré | **robustesse** |
| `773ee9a` | **Softmax** (remplace sigmoid) | **IA** |
| `b302b63` | DB par vidéo + health check Rust | **persistance** |

#### Phase 5 — Finalisation (16 juillet)
- Documentation technique + rapport final
- 6 diagrammes matplotlib
- 3 présentations PPTX

### 20.3 Évolution des modèles de race

```
HSV heuristique  ──❌ trop imprécis──►
    CLIP ViT-B/32  ──❌ sigmoid plat──►
        SigLIP-2 So400m + softmax  ──✅ état de l'art──►  FINAL
```

Chaque itération a résolu une limitation de la précédente, documentée dans les commits.

---

## 21. Résultats et métriques

### 21.1 Performance temps-réel

| Métrique | Valeur | Objectif | Statut |
|---|---|---|---|
| FPS moyen (M1 Pro) | **23-26** | 15 | ✅ Dépassé |
| Latence détection YOLO | ~12 ms | < 30 ms | ✅ |
| Latence Re-ID (batch) | ~5 ms async | < 10 ms | ✅ |
| Latence breed (SigLIP-2) | ~22 ms | < 50 ms | ✅ |
| Latence comportement (Numba) | ~0.1 ms | < 1 ms | ✅ |
| Latence annotation ROI | ~9.5 ms | < 15 ms | ✅ |
| Latence JPEG encode | ~3 ms off-thread | < 5 ms | ✅ |
| **Total par frame** | **~38 ms** | < 67 ms (15 FPS) | ✅ |

### 21.2 Précision

| Métrique | Valeur | Méthode |
|---|---|---|
| Précision détection | ~92 % | Validation visuelle sur vidéos test |
| Précision Re-ID (seuil 0.70) | Haute | Pas de faux positif observé |
| Précision race (SigLIP-2) | Variable | Dépend de la confusabilité |
| Taux "Indéterminée" | ~30 % races fauve | Attendu (seuil honnêteté) |

### 21.3 Métriques de classification de race (SigLIP-2 + softmax)

Cas type (Normande vs Angus) :

| Avant (sigmoid) | Après (softmax τ=0.01) |
|---|---|
| Normande: 0.51 | **Normande: 0.63** |
| Angus: 0.49 | Angus: 0.19 |
| Margin: 0.02 ❌ | **Margin: 0.44** ✅ |

Le softmax produit des marges exploitables, le sigmoid était inutilisable.

### 21.4 Empreinte ressources

| Ressource | Valeur |
|---|---|
| Binaire Tauri | ~1.8 Mo |
| RAM worker Python | ~2-3 Go (modèles chargés) |
| RAM Tauri shell | ~50 Mo |
| Poids YOLOv26 MLX | 46 Mo |
| Poids DINOv2-small | ~85 Mo |
| Poids SigLIP-2 So400m | ~4.5 Go |
| DB par vidéo | ~10-20 Ko |

### 21.5 Lignes de code

| Composant | LOC |
|---|---|
| Python (core) | ~4700 |
| JavaScript (frontend) | ~750 |
| Rust (Tauri) | ~266 |
| CSS | ~700 |
| HTML | ~400 |
| Documentation markdown | ~1200 |
| **Total** | **~8000** |

### 21.6 Couverture fonctionnelle

| Objectif initial | Statut |
|---|---|
| Détection + segmentation | ✅ |
| Tracking intra-vidéo | ✅ |
| Re-ID cross-vidéo | ✅ |
| Classification de race | ✅ |
| Interface utilisateur | ✅ |
| ≥ 15 FPS | ✅ (23-26) |

| Objectif additionnel | Statut |
|---|---|
| Classification comportements | ✅ |
| Dashboard analytique | ✅ |
| Heatmap spatiale | ✅ |
| Profils individuels | ✅ |
| App desktop cross-platform | ✅ |
| Noms persistants uniques | ✅ |
| Optimisation Apple Silicon | ✅ |
| Hot-reload paramètres | ✅ |
| Auto-récupération | ✅ |
| Documentation complète | ✅ |

**Bilan** : 6/6 objectifs initiaux + 10/10 objectifs additionnels = **16/16**.

---

## 22. Limites connues

### 22.1 Classification de race

- **Confusion races fauve** : Charolaise/Limousine/Salers/Blonde d'Aquitaine ont des robes similaires. SigLIP-2 hésite → ~30 % classés "Indéterminée".
- **Pas de fine-tuning** : le zero-shot est limité par la qualité des prompts. Un fine-tuning sur des images annotées améliorerait significativement.
- **Pas de races hybrides** : le système ne gère pas les croisements.

### 22.2 Re-ID

- **Bovins très similaires** : dans un troupeau de Charolaises toutes blanches, DINOv2 peut confondre deux individus.
- **Seuil 0.70 strict** : peut rater des matches légitimes si l'angle de vue diffère fortement entre vidéos.
- **Cap EMA 30** : après 30 updates, l'embedding est figé. Si le bovin change radicalement (croissance, robe), il peut être re-classé comme nouveau.

### 22.3 Conditions d'éclairage

- DINOv2 est robuste mais pas infaillible face aux ombres fortes
- HSV souffre sous éclairage artificiel (couleur non naturelle)
- LBP compense partiellement mais reste limité

### 22.4 Déploiement

- **Python requis côté utilisateur** : le bundle Tauri n'embarque pas Python
- **Modèles à télécharger** : SigLIP-2 (~4.5 Go) au premier lancement
- **MLX = macOS uniquement** : sur Windows/Linux, fallback PyTorch/CUDA (moins performant)
- **Pas de cross-compilation** : un build macOS → `.app` macOS uniquement

### 22.5 Streaming

- **JPEG polling** : bande passante plus élevée que MJPEG (headers HTTP répétés)
- **Pas de WebSocket** : latence de polling (1s pour stats, 40ms pour vidéo)
- **CSP désactivé** dans Tauri (nécessaire pour naviguer vers le worker HTTP)

### 22.6 Scalabilité

- **DB pickle** : pas adapté au-delà de ~1000 bovins (tout en mémoire)
- **Single-threaded Flask** : un seul worker vidéo à la fois
- **Pas de multi-caméras** : une seule source à la fois

---

## 23. Perspectives et améliorations futures

### 23.1 Améliorations IA

#### Fine-tuning de la classification de race
Collecter un dataset annoté de races françaises (~1000 images/race) et fine-tuner SigLIP-2 ou entraîner un EfficientNet spécialisé. Objectif : précision > 90 % sur toutes races.

#### Amélioration Re-ID
- **Ré-entraîner DINOv2** sur un dataset de bovins (transfer learning)
- **Attention mechanism** : pondérer les régions discriminantes (tête, pattern robe)
- **Triplet loss fine-tuning** : rapprocher les embeddings du même animal

#### Détection d'health
- **Body condition scoring** : estimer l'état d'embonpoint (morphologie)
- **Détection de boiterie** : analyse de la marche (asymétrie)
- **Détection de chaleur** : comportements spécifiques (monte)

### 23.2 Améliorations système

#### Multi-caméras
- Synchroniser plusieurs sources
- **Cross-camera Re-ID** : reconnaître un bovin vu par caméra A puis caméra B
- Vue 3D du cheptel (triangulation)

#### Edge deployment
- **Jetson Nano / Orin** : déploiement NVIDIA TensorRT
- **Raspberry Pi 5** : optimisation ARM (quantification INT8)
- **Déploiement cloud** : streaming RTSP distant

#### Real-time alerts
- Notification si bovin absent > 24h
- Alerte si comportement anormal (rué = stress/ danger)
- Comptage automatique pour inventaire

### 23.3 Améliorations UX

#### Mobile app
- Application compagnon (iOS/ Android) pour consultation à distance
- Notifications push

#### Mode offline
- Cache local des données analytiques
- Sync différée quand connexion retrouve

#### Rapports automatisés
- Export PDF quotidien/ hebdomadaire
- Statistiques de cheptel (âge moyen, distribution races)

### 23.4 Améliorations architecturales

#### Base de données scalable
Remplacer pickle par SQLite ou PostgreSQL pour :
- Support > 10 000 bovins
- Requêtes complexes (SQL)
- Multi-utilisateurs

#### Microservices
Découper en :
- Service de détection (GPU server)
- Service Re-ID (GPU server)
- Service API (CPU)
- Frontend (CDN)

#### Pipeline de données
- Apache Kafka pour les événements
- TimescaleDB pour les time-series (FPS, positions)
- Apache Spark pour l'analyse batch

### 23.5 Sécurité

- **Authentification** : JWT ou OAuth2 pour l'accès API
- **Chiffrement** : TLS entre worker et frontend
- **Audit log** : traçabilité des actions (reset DB, changement settings)
- **Sandboxing** : isoler le worker Python (Docker)

### 23.6 Capteurs additionnels

- **Microphone** : détection de toux (détresse respiratoire)
- **Capteur température** : fusion avec vision (détection fièvre)
- **GPS collier** : recalibrer le Re-ID visuel

---

## 24. Guide d'installation et de déploiement

### 24.1 Prérequis

#### macOS (recommandé pour MLX)
- macOS 13.0+ (Ventura)
- Python 3.10+
- Apple Silicon (M1/M2/M3) pour MLX
- ~10 Go d'espace (modèles)

#### Windows
- Windows 10/11
- Python 3.10+
- GPU NVIDIA recommandé (CUDA)
- Visual Studio Build Tools (pour Rust/Tauri)

#### Linux
- Ubuntu 22.04+
- Python 3.10+
- GPU NVIDIA recommandé (CUDA)
- `build-essential`, `libssl-dev`

### 24.2 Installation

```bash
# 1. Cloner le dépôt
git clone <repo-url> boeuf-tracker
cd boeuf-tracker

# 2. Créer le venv
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. (macOS) Installer MLX
pip install mlx mlx-vlm

# 5. Télécharger les poids YOLO
# Option A : YOLOv11 (PyTorch, universel)
# → téléchargement automatique au premier run

# Option B : YOLOv26 MLX (macOS, plus rapide)
# Placer yolo26s-seg.safetensors à la racine

# 6. Lancer le worker
python app.py --mlx --port 8100

# 7. Accéder à l'UI
open http://localhost:8100
```

### 24.3 Lancement en mode développement (avec Bun proxy)

```bash
./dev.sh
# Lance : worker Python (:8100) + Bun gateway (:8000)
# UI sur http://localhost:8000
```

### 24.4 Build de l'app desktop Tauri

```bash
# Prérequis : Rust + Tauri CLI
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install tauri-cli --version "^2.0"

# Build
./build.sh
# Output : src-tauri/target/release/bundle/
#   macOS :  Boeuf Tracker.app + .dmg
#   Win :    Boeuf Tracker.exe + .msi
#   Linux :  boeuf-tracker.AppImage + .deb
```

### 24.5 Variables CLI importantes

| Flag | Défaut | Description |
|---|---|---|
| `--source` | auto | Chemin vidéo ou index webcam |
| `--host` | 0.0.0.0 | Bind address |
| `--port` | 8100 | Port HTTP |
| `--yolo-model` | auto | yolo26s-seg.safetensors si présent |
| `--dino-model` | facebook/dinov2-small | Modèle Re-ID |
| `--threshold` | 0.70 | Seuil Re-ID normal |
| `--loop-threshold` | 0.55 | Seuil Re-ID post-rewind |
| `--loop-grace-frames` | 60 | Frames de grâce après rewind |
| `--max-updates` | 30 | Cap EMA par animal |
| `--conf` | 0.4 | Confiance YOLO minimum |
| `--device` | auto | mlx/cuda/mps/cpu |
| `--mlx` | off | Force MLX |
| `--imgsz` | 640 | Résolution YOLO |
| `--embed-every` | 10 | Frames entre re-embeds |
| `--skip-frames` | 0 | Skip max adaptatif |
| `--ui-dir` | web/public | Dossier UI statique |

### 24.6 Profiles de performance (GTX 1660 Ti)

Documentés dans `PERF_GUIDE_1660TI.md` :

| Profile | Modèle | imgsz | skip | embed_every | conf | FPS |
|---|---|---|---|---|---|---|
| **Max FPS** | yolo11n-seg | 416 | 20 | 2 | 0.45 | 35-45 |
| **Balanced** | yolo11s-seg | 640 | 10 | 1 | 0.40 | 20-28 |
| **Precision** | yolo11m-seg | 800 | 5 | 0 | 0.35 | 10-14 |

---

## 25. Structure du code source

### 25.1 Arborescence

```
boeuf-tracker/
├── app.py                      # Serveur Flask principal (758 lignes)
├── processor.py                # Pipeline vidéo legacy (988 lignes)
├── detector.py                 # Détection YOLO (410 lignes)
├── reid.py                     # Re-ID DINOv2 (318 lignes)
├── database.py                 # Base d'embeddings (240 lignes)
├── breed.py                    # Classification race SigLIP-2 (390 lignes)
├── names.py                    # Génération noms (187 lignes)
├── analytics.py                # Module analytics (347 lignes)
├── state.py                    # État global partagé
├── console.py                  # Logger coloré
├── capture.py                  # Gestion source vidéo
├── reid_worker.py              # Worker Re-ID async
├── watcher.py                  # Auto-reload
├── config.py                   # Configuration centralisée
├── bench_perf.py               # Benchmarks
├── main.py                     # Entrée CLI alternative
├── gradio_app.py               # UI Gradio (désactivée)
│
├── core/                       # Refactor OOP (nouveau)
│   ├── detector.py
│   ├── reid.py
│   ├── database.py
│   ├── processor.py
│   └── capture.py
├── utils/                      # Utils OOP (nouveau)
│   ├── state.py
│   ├── console.py
│   ├── names.py
│   └── reid_worker.py
├── models/                     # Modèles OOP (nouveau)
│   ├── animal.py
│   ├── analytics.py
│   └── breed.py
├── api/                        # Flask factory OOP (nouveau)
│   └── app.py
│
├── web/                        # Frontend vanilla
│   ├── src/server.ts           # Proxy Bun/Hono
│   ├── public/
│   │   ├── index.html
│   │   ├── app.js
│   │   ├── styles.css
│   │   └── splash.html
│   └── data/history.json
│
├── src-tauri/                  # App desktop Rust
│   ├── src/main.rs
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   ├── capabilities/default.json
│   └── icons/
│
├── docs/                       # Documentation
│   ├── PFE_individuel_documentation_technique.md
│   ├── PFE_individuel_rapport_final.md
│   ├── gen_diagrams.py
│   ├── gen_pptx.py
│   └── diagram_*.png           # 6 diagrammes
│
├── templates/                  # UI Flask legacy
├── static/                     # JS Flask legacy
├── frontend/                   # Next.js WIP (inactif)
│
├── requirements.txt
├── README.md
├── PERF_GUIDE_1660TI.md
├── dev.sh
├── build.sh
└── *.mp4, *.pt, *.pkl          # Données (gitignored)
```

### 25.2 Dual-structure legacy / OOP

Le codebase présente **deux structures parallèles** :

- **Modules root** (`app.py`, `processor.py`, `detector.py`...) : pipeline **legacy monolithique**, c'est le chemin d'exécution actuel via `app.py → processor.py → root modules`
- **Packages OOP** (`core/`, `utils/`, `models/`, `api/`) : refactor orienté objet en cours, pas encore le runtime

Cette dualité reflète une migration en cours. Les deux implémentations sont logiquement équivalentes.

### 25.3 Tests

Le projet n'a pas de suite de tests automatisés formelle. La validation repose sur :

- **Benchmarks** (`bench_perf.py`) : performance match, behavior, DINOv2
- **Videos de test** : 3 fichiers `.mp4` (5,7 Mo, 74,1 Mo, 7,9 Mo)
- **Validation visuelle** : inspection manuelle de l'UI
- **Endpoint `/api/bench`** : benchmark live YOLO11 vs YOLO26 vs DINOv2

---

## 26. Glossaire technique

| Terme | Définition |
|---|---|
| **Bounding box** | Rectangle délimitant un objet détecté `[x1, y1, x2, y2]` |
| **Segmentation d'instance** | Attribution pixel-par-pixel de chaque objet (vs bounding box) |
| **Mask** | Image binaire (H×W) où 1 = pixel de l'objet, 0 = fond |
| **IoU** | Intersection over Union = intersection / union de deux boxes |
| **Tracker** | Algorithme maintenant un ID entre frames (ByteTrack, IoU tracker) |
| **Embedding** | Représentation vectorielle dense d'une image (ex. : 464-dim) |
| **Cosine similarity** | `dot(a,b) / (norm(a)*norm(b))`, ∈ [-1, 1] |
| **Re-ID** | Ré-identification : reconnaître le même individu à travers vidéos |
| **Zero-shot** | Classification sans images d'entraînement de la classe |
| **VLM** | Vision-Language Model (ex. : CLIP, SigLIP) |
| **Softmax** | Fonction convertissant un vecteur en distribution de probabilité |
| **Sigmoid** | Fonctionbornante indépendante par classe (σ(x) = 1/(1+e^-x)) |
| **Temperature** | Paramètre τ contrôlant la "sharpness" du softmax |
| **Margin** | Différence top1 - top2 dans une distribution |
| **DINOv2** | Modèle self-supervised de Meta (2023) produisant des embeddings |
| **SigLIP-2** | VLM de Google (2025), état de l'art zero-shot |
| **MLX** | Framework ML d'Apple pour Silicon (Metal GPU natif) |
| **MPS** | Metal Performance Shaders (backend PyTorch sur Metal) |
| **FP16** | Flottant demi-précision (16-bit), 2× plus rapide que FP32 |
| **EMA** | Exponential Moving Average : `new = (1-α)*old + α*input` |
| **Pickle** | Format de sérialisation binaire Python |
| **BLAS** | Basic Linear Algebra Subprograms (opérations matricielles optimisées) |
| **Numba JIT** | Just-In-Time compiler Python → LLVM |
| **LBP** | Local Binary Pattern (descripteur de texture) |
| **HSV** | Hue Saturation Value (espace couleur) |
| **COCO** | Common Objects in Context (dataset, 80 classes dont vache=19) |
| **ByteTrack** | Tracker multi-objets basé IoU (Ultralytics) |
| **Sidecar** | Processus enfant lancé et supervisé par un parent (Tauri) |
| **Hot-reload** | Modification de paramètres sans redémarrage |
| **Splash screen** | Écran de chargement au démarrage |
| **Jpeg polling** | Polling d'images JPEG (vs streaming MJPEG) |
| **CSP** | Content Security Policy (en-tête HTTP de sécurité) |

---

## 27. Références

### 27.1 Modèles et papers

1. **YOLOv11** — Ultralytics (2024). *Documentation officielle*. https://docs.ultralytics.com/models/yolo11/
2. **YOLOv26 MLX** — Adaptation Apple Silicon de YOLO. https://github.com/HuertsMLX/yolo26mlx
3. **DINOv2** — Oquab et al. (Meta AI, 2023). *DINOv2: Learning Robust Visual Features without Supervision*. https://arxiv.org/abs/2304.07193
4. **SigLIP** — Zhai et al. (Google, 2023). *Sigmoid Loss for Language Image Pre-Training*. https://arxiv.org/abs/2303.15343
5. **SigLIP-2** — Tschannen et al. (Google, 2025). *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Foundations*. https://arxiv.org/abs/2502.14786
6. **ByteTrack** — Zhang et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. https://arxiv.org/abs/2110.06864
7. **CLIP** — Radford et al. (OpenAI, 2021). *Learning Transferable Visual Models From Natural Language Supervision*. https://arxiv.org/abs/2103.00020

### 27.2 Technologies

8. **MLX** — Apple. *MLX: An Array Framework for Machine Learning on Apple Silicon*. https://github.com/ml-explore/mlx
9. **PyTorch** — Paszke et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. https://pytorch.org/
10. **Transformers (HuggingFace)** — Wolf et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. https://huggingface.co/docs/transformers
11. **Ultralytics** — YOLO implementation. https://github.com/ultralytics/ultralytics
12. **OpenCV** — Bradski (2000). *The OpenCV Library*. https://opencv.org/
13. **Flask** — Pallets Projects. https://flask.palletsprojects.com/
14. **Tauri** — Tauri Programme. *Tauri v2 Documentation*. https://v2.tauri.app/
15. **Numba** — Lam et al. (2015). *Numba: A LLVM-based Python JIT Compiler*. https://numba.pydata.org/
16. **Chart.js** — Chart.js.org. *Simple yet flexible JavaScript charting*. https://www.chartjs.org/
17. **Bun** — Oven-sh. *Bun: Incredibly fast JavaScript runtime*. https://bun.sh/
18. **Hono** — Hono.dev. *Ultrafast Web Framework for the Edges*. https://hono.dev/

### 27.3 Dataset et classes

19. **COCO Dataset** — Lin et al. (2014). *Microsoft COCO: Common Objects in Context*. https://cocodataset.org/
    - Classe 19 : "cow" (vache)

### 27.4 Outils

20. **python-pptx** — Pour la génération des présentations. https://python-pptx.readthedocs.io/
21. **matplotlib** — Pour la génération des diagrammes. https://matplotlib.org/
22. **ngrok** — Tunnels HTTP sécurisés. https://ngrok.com/
23. **Google Colab** — Exécution GPU gratuite. https://colab.research.google.com/

---

## Conclusion

Le projet **Boeuf Tracker** démontre la viabilité d'une approche vision par ordinateur pour la surveillance d'élevage bovin, en combinant quatre technologies d'IA état de l'art (YOLOv26 MLX, DINOv2, SigLIP-2, Numba) dans un système cohérent et performant.

Les **résultats dépassent les objectifs initiaux** : 23-26 FPS (vs 15 visés), 16 fonctionnalités livrées (vs 6 prévues), application desktop cross-platform, dashboard analytique complet, et classification de race zero-shot sans données d'entraînement.

La **philosophie d'honnêteté** (seuil de marge 0.15 pour refuser de deviner une race incertaine) et la **persistance d'identité** (compteur global jamais réutilisé, DB par vidéo) témoignent d'une conception prudente et robuste.

Les **perspectives** (fine-tuning race, multi-caméras, edge deployment, capteurs additionnels) ouvrent la voie à un produit commercialisable, tout en restant fondées sur une architecture éprouvée et documentée.

Ce projet illustre l'intégration réussie de technologies IA modernes dans un cas d'usage concret et utile, avec une attention particulière à la performance, la robustesse, et l'expérience utilisateur — des compétences centrales du génie électrique contemporain.

---

*Rapport généré le 22 juillet 2026 — 55 commits, ~8000 lignes de code, 6 jours de développement actif.*

**Fin du rapport.**
