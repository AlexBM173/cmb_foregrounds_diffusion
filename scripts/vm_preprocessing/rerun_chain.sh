#!/bin/bash
# NB01 (1e13 floor) -> NB02 (2.03e14 file-unit cut = paper's 3e14 M_sun) -> NB03.
# Launched Sat 2026-07-04 after the units gate: totm500 is M_sun/h.
set -eo pipefail
cd ~/cmb_foregrounds_diffusion
source ~/venv/bin/activate

python nb01_run.py 2>&1 | tee logs/nb01_1e13.log

# Preserve the first-pass patches (over-cut 3e14-file-unit mask) as fallback
if [ -d data/low_pass/2mJy ] && [ ! -d data/low_pass/2mJy_fallback_overcut ]; then
    mv data/low_pass/2mJy data/low_pass/2mJy_fallback_overcut
fi

export HALO_CAT=$HOME/cmb_foregrounds_diffusion/data/halo_catalogue/halo_catalogue_m500gt1e13.npz
export M500C_THRESHOLD=2.03e14
python nb02_run.py 2>&1 | tee logs/nb02.log
python nb03_run.py 2>&1 | tee logs/nb03.log
echo "RERUN CHAIN COMPLETE"
