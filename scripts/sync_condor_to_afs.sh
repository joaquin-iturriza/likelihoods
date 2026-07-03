#!/usr/bin/env bash
# Deploy the in-repo Condor job infrastructure (condor/) to the AFS submission
# side. Source of truth is EOS (this repo); AFS is a deploy target because
# HTCondor won't submit from EOS.
#
# Syncs ONLY the source-of-truth files (generators + hand-written templates).
# It never touches AFS-side runtime dirs (jobs/, subs/, error/, log/, output/,
# runs/, sweeps/) — those are generated on AFS and must not be clobbered.
#
# Usage:  scripts/sync_condor_to_afs.sh [--dry-run]
set -euo pipefail

REPO="/eos/home-j/joiturri/likelihoods"
AFS="/afs/cern.ch/user/j/joiturri/likelihoods"
SRC="$REPO/condor/"

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="--dry-run"

if [ ! -d "$SRC" ]; then
  echo "ERROR: $SRC does not exist" >&2; exit 1
fi
if [ ! -d "$AFS" ]; then
  echo "ERROR: AFS submission dir $AFS not reachable (are you on lxplus with a token?)" >&2; exit 1
fi

# --ignore-existing is NOT used: we want generator edits to propagate. But we do
# NOT delete on the AFS side (no --delete), so runtime dirs/logs are preserved.
rsync -av $DRY \
  --exclude '._*' --exclude '.DS_Store' --exclude '__pycache__' \
  "$SRC" "$AFS/"

echo
echo "Synced condor/ -> $AFS"
echo "Next: cd $AFS && python <generator>.py   (then confirm before condor_submit)"
