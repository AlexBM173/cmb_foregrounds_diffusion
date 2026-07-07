#!/bin/bash
# 4-channel preprocessing: reuse the CIB/tSZ masked FITS from the original run,
# mask the new kSZ/kappa channels, then extract 4-channel patches.
#
# Prereqs on the VM:
#   * data/cib_150_masked.fits, data/tsz_150_masked.fits  (from nb02_run.py)
#   * raw CIB FITS in ~/agora_data (for rebuilding the point-source mask)
#   * raw kSZ + kappa FITS in ~/agora_data (from Globus)
#   * data/halo_catalogue/halo_catalogue_m500gt1e13.npz  (from nb01_run.py)
#
# CONFIRM from the FITS headers before running (see nb02b docstring):
#   KSZ_FILE / KSZ_INPUT_UNIT   — exact kSZ filename + whether uK or dT/T
#   KAPPA_FILE / KAPPA_FIELD    — exact kappa filename + convergence column index
set -eo pipefail
cd ~/cmb_foregrounds_diffusion
source ~/venv/bin/activate
mkdir -p logs

# --- fill these in from `fitsheader ~/agora_data/<file> | grep -Ei 'TTYPE|TUNIT|NSIDE'` ---
export KSZ_FILE="${KSZ_FILE:-*ksz*lensed*.fits}"
export KSZ_INPUT_UNIT="${KSZ_INPUT_UNIT:-uK}"          # uK | dimensionless
export KAPPA_FILE="${KAPPA_FILE:-*cmbkappa*.fits}"
export KAPPA_FIELD="${KAPPA_FIELD:-0}"                  # column holding convergence
export M500C_THRESHOLD="${M500C_THRESHOLD:-2.03e14}"    # must match the CIB/tSZ run
export HALO_CAT="${HALO_CAT:-$HOME/cmb_foregrounds_diffusion/data/halo_catalogue/halo_catalogue_m500gt1e13.npz}"

python scripts/vm_preprocessing/nb02b_mask_ksz_kappa.py 2>&1 | tee logs/nb02b.log
python scripts/vm_preprocessing/nb03b_extract_4ch.py     2>&1 | tee logs/nb03b.log
echo "4-CHANNEL PREPROCESSING COMPLETE"
