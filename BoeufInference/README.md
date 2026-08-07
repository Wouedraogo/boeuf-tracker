# BoeufInference

Petite app macOS SwiftUI pour tester tes 4 modèles CoreML (YOLO26 + CBVD5) sur
un dossier de vidéos. Utilise **CoreML + Compute Units `.all`** → l'ANE est
utilisé automatiquement quand Apple juge qu'il gagne.

## Modèles embarqués

- `boeuf_cbvd5_lr001` (détection)
- `boeuf_cbvd5_seg` (segmentation — pour l'instant on n'affiche que les boîtes)
- `boeuf_yolo26s_640`
- `boeuf_yolo26s_832`

Tous exportés par Ultralytics avec NMS fusée : entrée image 640×640 RGB, sortie
`[1, 300, 6]` = `(x1, y1, x2, y2, score, class)` en pixels du modèle.

## Ouvrir / lancer

```bash
open BoeufInference/BoeufInference.xcodeproj
```

Ou en ligne de commande :

```bash
cd BoeufInference
xcodebuild -project BoeufInference.xcodeproj -scheme BoeufInference -configuration Debug \
  -destination 'platform=macOS' build \
  CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO
open ~/Library/Developer/Xcode/DerivedData/BoeufInference-*/Build/Products/Debug/BoeufInference.app
```

## Utilisation

1. Sidebar → choisir le modèle.
2. Sidebar → **Choisir…** un dossier contenant `.mp4 .mov .m4v .avi .mkv`.
3. Cliquer une vidéo → lecture + boîtes affichées en direct.
4. Slider **Confiance** pour filtrer.

Le status bar en bas affiche : modèle, inférences/s, nb de détections courantes.

## Régénérer le projet

Le `.xcodeproj` est généré par [xcodegen](https://github.com/yonaskolb/XcodeGen)
à partir de `project.yml` :

```bash
cd BoeufInference
xcodegen generate
```

## Prochaine étape

Une fois la détection validée sur vidéo, on branche le tracking
(ByteTrack/BoT-SORT côté Swift, ou passerelle vers le pipeline Python existant).
