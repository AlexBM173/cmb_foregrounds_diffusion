#!/usr/bin/env bash
# =============================================================================
# v5_4ch_zscore_2mJy_a100 — Colab SAMPLING launcher
#
# 4-channel DDPM (CIB, tSZ, kSZ, CMB-lensing κ), dim=96. One-shot: clone the
# feat/v5-4channel branch, pull the final checkpoint from the GCS run dir, run
# `run.py sample` (640 maps, full 1000-step ancestral DDPM), 4-CHANNEL range
# check, sync samples + logs back to GCS, drop SAMPLING_DONE.txt.
#
# `run.py sample` auto-selects the latest model-*.pt in the run dir and threads
# --channels 4 --dim 96 from config/v5_4ch.yaml, so no CLI surgery is needed.
#
# PREREQUISITE: training has produced runs/$RUN/checkpoints/model-20.pt in GCS
#   (TRAIN_DONE.txt present). The periodic sync in train_v5.sh keeps that path
#   populated even without a clean finish.
#
# Paste into a Colab cell (A100 runtime):
#   from google.colab import auth; auth.authenticate_user()
#   !gcloud config set project cmb-diffusion-migration -q
#   !gcloud -q storage cp gs://cmb-diffusion-artifacts-alexbm173/colab/sample_v5.sh /content/ \
#       && bash /content/sample_v5.sh
# =============================================================================
set -euo pipefail

BUCKET=gs://cmb-diffusion-artifacts-alexbm173
RUN=v5_4ch_zscore_2mJy_a100
BRANCH=feat/v5-4channel
CONFIG=config/v5_4ch.yaml
CKPT=model-20.pt   # final milestone (step 100000)

# --- status markers (visible from outside Colab) ---
trap 'date -u +"%Y-%m-%dT%H:%M:%SZ failed at line $LINENO" | gcloud -q storage cp - "$BUCKET/runs/$RUN/SAMPLING_FAILED.txt" || true' ERR
gcloud -q storage rm "$BUCKET/runs/$RUN/SAMPLING_DONE.txt" "$BUCKET/runs/$RUN/SAMPLING_FAILED.txt" 2>/dev/null || true
date -u +"%Y-%m-%dT%H:%M:%SZ" | gcloud -q storage cp - "$BUCKET/runs/$RUN/SAMPLING_STARTED.txt"

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

# Pinned lib version — same as v4/training, required for checkpoint arch compat.
pip install -q -e . "denoising-diffusion-pytorch==2.2.6"

# --- checkpoint: pull the final model from the GCS run dir ---
#   (checkpoints/$RUN/ only exists after a clean TRAIN_DONE; the run-dir path is
#    kept current by train_v5.sh's periodic sync, so it is the robust source.)
mkdir -p "runs/$RUN/checkpoints" "runs/$RUN/samples" "runs/$RUN/logs"
if [ ! -f "runs/$RUN/checkpoints/$CKPT" ]; then
  echo "== pulling checkpoint $CKPT (~1.3 GB, dim=96) =="
  gcloud -q storage cp "$BUCKET/runs/$RUN/checkpoints/$CKPT" "runs/$RUN/checkpoints/" \
    || { echo "ERROR: $CKPT not in GCS run dir — has training finished (TRAIN_DONE)?"; exit 1; }
fi

# --- sample: 640 maps, 1000-step ancestral, batch 16; run.py threads dim/channels ---
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="runs/$RUN/logs/sample_${STAMP}.log"
echo "== sampling: 640 maps, 4-channel, 1000-step ancestral, batch 16 — log: $LOG =="
python run.py sample --config "$CONFIG" 2>&1 | tee "$LOG"

# --- 4-channel range check (z-score units; must NOT be pinned to [-1, 1]) ---
echo "== range check (4 channels; z-score units) =="
python - <<'PY'
import numpy as np
x = np.load("runs/v5_4ch_zscore_2mJy_a100/samples/samples_v5_4ch_zscore_2mJy_a100.npy")
print("samples:", x.shape, x.dtype)
chan_first = x.ndim == 4 and x.shape[1] == 4
assert (x.shape[1] if chan_first else x.shape[-1]) == 4, f"expected 4 channels, got {x.shape}"
for i, name in enumerate(["CIB", "tSZ", "kSZ", "kappa"]):
    c = x[:, i] if chan_first else x[..., i]
    print(f"{name:>5}: min {c.min():+.2f}  max {c.max():+.2f}  "
          f"mean {c.mean():+.3f}  std {c.std():.3f}")
    assert not (abs(c.min() + 1) < 1e-3 and abs(c.max() - 1) < 1e-3), \
        f"{name} pinned to [-1, 1] — clamp regression, do NOT use these samples"
print("range check OK — 4 channels, unclamped")
PY

# --- sync to GCS ---
echo "== syncing to GCS =="
gcloud -q storage rsync --recursive "runs/$RUN/samples" "$BUCKET/runs/$RUN/samples"
gcloud -q storage rsync --recursive "runs/$RUN/logs" "$BUCKET/runs/$RUN/logs"
date -u +"%Y-%m-%dT%H:%M:%SZ" | gcloud -q storage cp - "$BUCKET/runs/$RUN/SAMPLING_DONE.txt"

echo "== ALL DONE — samples at $BUCKET/runs/$RUN/samples/ =="
echo "   next: python run.py evaluate --config $CONFIG   (CPU-cheap, can run locally)"
echo "Terminate the runtime now (Runtime > Disconnect and delete runtime) to stop compute units."
