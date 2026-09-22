#!/bin/bash
# likelihoods: the ATLAS yield/nLL datasets in data/ (not regenerable, not in git) and a pdflatex for usetex plots.
# Contract (run by `site env`, checked by `site pick`): with sites/activate.sh sourced,
#   bash sites/setup.sh            prepare this site for the project (idempotent)
#   bash sites/setup.sh --verify   fast, read-only: exit 0 iff runnable, one status line
set -u

verify() {
  local miss=()
  n=$(ls data/*.npy 2>/dev/null | wc -l); [ "$n" -gt 0 ] || miss+=("data/*.npy (copy from the lxplus checkout /eos/user/j/joiturri/likelihoods/data)")
  if ! command -v pdflatex >/dev/null 2>&1; then
    (command -v module >/dev/null 2>&1 && module -t avail 2>&1 | grep -q '^texlive') || miss+=("pdflatex (base_plots uses usetex)")
  fi
  if [ ${#miss[@]} -gt 0 ]; then echo "missing: ${miss[*]}"; return 1; fi
  echo "ok: $n data files, pdflatex available"
}
[ "${1:-}" = "--verify" ] && { verify; exit $?; }
mkdir -p data
verify || true      # the data is copied, never generated; pdflatex is the site's
exit 0
