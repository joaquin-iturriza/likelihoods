#!/bin/bash
#SBATCH --job-name=stage_rafal_unc
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# CPU job, needs internet (CC-IN2P3 compute nodes have it): R. Maselek's final 2018-16 scans
# with signal uncertainty (1911.12606-uncertainty-final-results.tar.gz, Google Drive), their
# cleaned EWKinos and sleptons tables, and the same tables without the rows that
# tools/label_check.py flags (suffix -lc). Leaves existing files alone.
#   site submit ccin2p3 likelihoods scripts/stage_rafal_unc.sh
set -euo pipefail
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"

ID=1txM6g2cVidKLfwHWX8ERZOQph6FK9gH2
TGZ_MD5=0dc28e9edd1932062df603c40520a1aa
RAW="$DATA_DIR/raw_1911.12606-uncertainty-final"
TGZ="$RAW/1911.12606-uncertainty-final-results.tar.gz"
mkdir -p "$RAW"
if [ ! -f "$TGZ" ] || [ "$(md5sum < "$TGZ" | cut -d' ' -f1)" != "$TGZ_MD5" ]; then
    curl -sSL -o "$TGZ.part" "https://drive.usercontent.google.com/download?id=$ID&export=download&confirm=t"
    mv "$TGZ.part" "$TGZ"
fi
[ "$(md5sum < "$TGZ" | cut -d' ' -f1)" = "$TGZ_MD5" ] || { echo "tarball md5 mismatch"; exit 1; }

# stem  member-dir  csv-md5
while read -r STEM DIR MD5; do
    for EXT in csv json; do
        [ -f "$RAW/$STEM.$EXT" ] || tar xzf "$TGZ" -C "$RAW" --strip-components=4 \
            "1911.12606-uncertainty/$DIR/training_data/cleaned/$STEM.$EXT" 2>/dev/null
    done
    [ "$(md5sum < "$RAW/$STEM.csv" | cut -d' ' -f1)" = "$MD5" ] || { echo "$STEM.csv md5 mismatch"; exit 1; }
    OUT="$DATA_DIR/$STEM-lc.npy"
    if [ -f "$OUT" ]; then echo "$OUT exists, left alone"; continue; fi
    python tools/label_check.py "$RAW/$STEM.csv" "$RAW/$STEM-lc_flags" --drop "$OUT"
    cp -n "$RAW/$STEM.json" "$DATA_DIR/$STEM-lc.json"
done <<'EOF'
1911.12606-EWKinos-merged-big-tuned2-corr-nll300delta100-z5 EWKinos 1811e65263b5c61047bbacc65ad6d414
1911.12606-sleptons-merged-big-tuned2-corr-nll300delta150 Sleptons a7ca2b9d7b187ebb2dd3a0c2e0de5029
EOF
ls -la "$DATA_DIR"/*-lc.*
