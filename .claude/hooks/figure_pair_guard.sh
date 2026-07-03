#!/usr/bin/env bash
# Stop hook — every figure must exist as BOTH .png and .pdf (matching basename).
#
# Why: the user always wants each figure in png AND pdf; relying on the model to
# remember is unreliable. This checks at end of turn and BLOCKS the stop (exit 2)
# if a figure touched THIS session is missing its counterpart, so the model
# produces the missing format before finishing.
#
# Figures are gitignored (*.png and *.pdf), so this works off the filesystem + an
# mtime marker (.claude/.figpair_last): ONLY figures created/modified since the
# last check are inspected. Legacy single-format figures already on disk are never
# retroactively flagged.
#
# Heavy/auto-output trees are pruned for speed.
#
# Escape hatch: list basenames or repo-relative paths to skip in
# .claude/figure_pair_ignore.txt (one per line).
set -uo pipefail
REPO="/eos/home-j/joiturri/likelihoods"
MARKER="$REPO/.claude/.figpair_last"
IGNORE="$REPO/.claude/figure_pair_ignore.txt"
cd "$REPO" 2>/dev/null || exit 0

# First run: establish a baseline; don't retroactively flag existing figures.
if [ ! -e "$MARKER" ]; then
  : > "$MARKER"; exit 0
fi

# Figures changed since the last successful check.
mapfile -t figs < <(
  find . \( -path ./runs -o -path ./sweeps -o -path ./data -o -path ./models_onnx \
            -o -path ./models_onnx_deprecated -o -path ./IntrinsicDimDeep \
            -o -path ./validations -o -path ./.git -o -path './wt-*' \
            -o -path '*/scratchpad' -o -path ./scratch \) -prune \
       -o \( -name '*.png' -o -name '*.pdf' \) -newer "$MARKER" -print 2>/dev/null \
  | sed 's|^\./||'
)
[ ${#figs[@]} -eq 0 ] && { : > "$MARKER"; exit 0; }

missing=""
for f in "${figs[@]}"; do
  base_name=$(basename "$f")
  if [ -f "$IGNORE" ] && { grep -qxF "$f" "$IGNORE" 2>/dev/null || grep -qxF "$base_name" "$IGNORE" 2>/dev/null; }; then
    continue
  fi
  stem=${f%.*}
  [ -e "$stem.png" ] || missing="$missing|$stem.png"
  [ -e "$stem.pdf" ] || missing="$missing|$stem.pdf"
done

missing=$(printf '%s' "$missing" | tr '|' '\n' | sed '/^$/d' | sort -u)
if [ -n "$missing" ]; then
  {
    echo "BLOCKED by figure_pair_guard: every figure must be saved as BOTH .png and .pdf."
    echo "Missing counterpart(s) for figures changed this turn:"
    printf '  %s\n' $missing
    echo "Generate the missing format(s) — plot scripts should emit both — then finish."
    echo "If a file legitimately has no counterpart, add its path/basename to .claude/figure_pair_ignore.txt."
  } >&2
  exit 2   # leave MARKER unchanged so the same window is re-checked after the fix
fi

: > "$MARKER"   # all paired — advance the baseline
exit 0
