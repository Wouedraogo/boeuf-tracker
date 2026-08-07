#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# download_cbvd5.sh — Recupere le dataset public CBVD-5 depuis Kaggle
#
# CBVD-5 (Fandaoerji et al., Nature Scientific Reports 2024) :
#   96 h de barn / 107 vaches / 5 comportements (standing, lying,
#   foraging, rumination, drinking) / 687 clips 10 s + 206 100 frames.
#
# Sortie : external/cbvd5/  (extrait, pret pour training/train_cbvd5.py)
# ═══════════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")/.."

DEST="external/cbvd5"
mkdir -p external

if [ -d "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then
    echo "[cbvd5] Deja present dans $DEST — supprime le dossier pour re-telecharger."
    exit 0
fi

# ─── Prerequis : Kaggle CLI + token API ─────────────────────────────
if ! command -v kaggle >/dev/null 2>&1; then
    echo "[cbvd5] Kaggle CLI absent. Installation :"
    echo "  pip install kaggle"
    echo
    echo "Puis va sur https://www.kaggle.com/settings/account → 'Create New API Token'"
    echo "→ telecharge kaggle.json → depose-le dans ~/.kaggle/kaggle.json"
    echo "  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "  chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "[cbvd5] Token Kaggle absent ($HOME/.kaggle/kaggle.json)."
    echo "  Va sur https://www.kaggle.com/settings/account → Create New API Token"
    exit 1
fi

# ─── Telechargement + extraction ────────────────────────────────────
echo "[cbvd5] Telechargement depuis Kaggle (~2 GB, prend quelques minutes)..."
kaggle datasets download -d fandaoerji/cbvd-5cow-behavior-video-dataset -p external/

ARCHIVE=$(ls -t external/cbvd*.zip 2>/dev/null | head -1)
if [ -z "$ARCHIVE" ]; then
    echo "[cbvd5] Aucun .zip trouve apres telechargement — verifie kaggle CLI."
    exit 1
fi

echo "[cbvd5] Extraction vers $DEST..."
mkdir -p "$DEST"
unzip -q -o "$ARCHIVE" -d "$DEST"
rm -f "$ARCHIVE"

echo
echo "[cbvd5] OK. Contenu de $DEST :"
ls -1 "$DEST" | head -20
echo
echo "Lance l'entrainement :"
echo "  .venv/bin/python training/train_cbvd5.py --data $DEST --epochs 20"
