#!/usr/bin/env bash
# =============================================================================
# v5_4ch_zscore_2mJy_a100 — Colab TRAINING launcher (resumable)
#
# 4-channel DDPM (CIB, tSZ, kSZ, CMB-lensing κ), dim=96. Training is ~2× v4
# (~50-60h) and exceeds a single Colab session, so this script is RESUMABLE:
# re-run the SAME cell after any disconnect and it auto-detects the latest
# checkpoint in GCS and continues. A background loop pushes new checkpoints to
# GCS every 10 min so a crash loses at most the current (unfinished) milestone.
#
# PREREQUISITE: the feat/v5-4channel branch must be pushed to origin
#   (git push -u origin feat/v5-4channel) — this script clones it from GitHub.
#
# Paste into a Colab cell (A100 runtime):
#   from google.colab import auth; auth.authenticate_user()
#   !gcloud config set project cmb-diffusion-migration -q
#   !gcloud -q storage cp gs://cmb-diffusion-artifacts-alexbm173/colab/train_v5.sh /content/ \
#       && bash /content/train_v5.sh
# =============================================================================
set -euo pipefail

BUCKET=gs://cmb-diffusion-artifacts-alexbm173
RUN=v5_4ch_zscore_2mJy_a100
BRANCH=feat/v5-4channel
CONFIG=config/v5_4ch.yaml
PATCHES=$BUCKET/patches/v5_4ch
PATCH_DIR=data/low_pass/2mJy

# --- status markers (visible from outside Colab) ---
trap 'date -u +"%Y-%m-%dT%H:%M:%SZ failed at line $LINENO" | gcloud -q storage cp - "$BUCKET/runs/$RUN/TRAIN_FAILED.txt" || true' ERR
date -u +"%Y-%m-%dT%H:%M:%SZ" | gcloud -q storage cp - "$BUCKET/runs/$RUN/TRAIN_STARTED.txt"

echo "== GPU =="; nvidia-smi -L

# --- code: clone + checkout the 4-channel branch ---
cd /content
[ -d cmb_foregrounds_diffusion ] || \
  git clone --quiet https://github.com/AlexBM173/cmb_foregrounds_diffusion.git
cd cmb_foregrounds_diffusion
git fetch --quiet origin
git checkout --quiet "$BRANCH"
git pull --quiet --ff-only origin "$BRANCH" || true
echo "== repo at $(git rev-parse --short HEAD) ($BRANCH) =="

# Pinned lib version — same as v4, required for checkpoint architecture compat.
pip install -q -e . "denoising-diffusion-pytorch==2.2.6"

# --- training patches: GCS -> local (train.py loads the 4 channel .npy here) ---
mkdir -p "$PATCH_DIR"
gcloud -q storage rsync "$PATCHES" "$PATCH_DIR"
NCH=$(ls -1 "$PATCH_DIR"/*zscore*lp.npy 2>/dev/null | wc -l)
echo "== patches synced: $NCH channel files in $PATCH_DIR =="
[ "$NCH" -ge 4 ] || { echo "expected >=4 channel files, got $NCH"; exit 1; }

# --- resume detection: pull any existing run dir (checkpoints) from GCS ---
mkdir -p "runs/$RUN/checkpoints" "runs/$RUN/logs"
gcloud -q storage rsync --recursive "$BUCKET/runs/$RUN" "runs/$RUN" 2>/dev/null || true
if ls runs/"$RUN"/checkpoints/model-*.pt >/dev/null 2>&1; then
  LATEST=$(ls runs/"$RUN"/checkpoints/model-*.pt | sort -t- -k2 -n | tail -1)
  echo "== RESUMING from $LATEST =="
  export RESUME=1
else
  echo "== fresh start (no checkpoints in GCS) =="
  export RESUME=0
fi

# --- background checkpoint sync every 10 min (checkpoints written every 5000
#     steps; this makes a disconnect recoverable to the last full milestone) ---
( while true; do
    sleep 600
    gcloud -q storage rsync --recursive "runs/$RUN/checkpoints" "$BUCKET/runs/$RUN/checkpoints" || true
    gcloud -q storage rsync --recursive "runs/$RUN/logs" "$BUCKET/runs/$RUN/logs" || true
  done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true' EXIT

# --- train (single A100; Accelerator handles device placement) ---
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="runs/$RUN/logs/train_${STAMP}.log"
echo "== training (RESUME=$RESUME) — streaming to $LOG =="
python run.py train --config "$CONFIG" 2>&1 | tee "$LOG"

# --- final sync: full run dir + copy the newest checkpoint to checkpoints/<run>/
#     (v4 layout, so the sampling launcher finds it there) ---
kill "$SYNC_PID" 2>/dev/null || true
gcloud -q storage rsync --recursive "runs/$RUN" "$BUCKET/runs/$RUN"
FINAL=$(ls runs/"$RUN"/checkpoints/model-*.pt | sort -t- -k2 -n | tail -1)
gcloud -q storage cp "$FINAL" "$BUCKET/checkpoints/$RUN/$(basename "$FINAL")"
date -u +"%Y-%m-%dT%H:%M:%SZ" | gcloud -q storage cp - "$BUCKET/runs/$RUN/TRAIN_DONE.txt"

echo "== TRAINING COMPLETE — final checkpoint: $FINAL =="
echo "   run dir:     $BUCKET/runs/$RUN/"
echo "   final model: $BUCKET/checkpoints/$RUN/$(basename "$FINAL")"
echo "Terminate the runtime now (Runtime > Disconnect and delete runtime) to stop compute units."
