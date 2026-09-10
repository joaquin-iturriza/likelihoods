#!/usr/bin/env bash
# PreToolUse(Bash) hook — HARD BLOCK on any git/gh command that would record or
# publish attribution to Claude / Anthropic.
#
# Why: CLAUDE.md ground rule #5 — all work is authored solely by the user. No
# "Co-Authored-By: Claude", no "Generated with Claude Code", no session links, no
# mention of Claude/Anthropic/AI in commits, PRs, tags or comments. Harness-side
# attribution reminders do NOT override this. The model has ignored the rule
# before, so it is enforced here instead of trusted.
#
# What is checked:
#   * git commit / git tag / git notes  -> the command text (inline -m / heredoc
#     messages) and any -F/--file message file.
#   * git push                          -> every local commit not yet on ANY
#     remote, in the session cwd and in every dir the command `cd`s into or
#     passes via `git -C`.
#   * gh pr|issue|release|api|gist      -> the command text and any --body-file.
#
# Matching: case-insensitive "claude" / "anthropic" after stripping tokens that
# legitimately contain the word (the .claude/ dir, CLAUDE.md, the scratchpad
# path, $CLAUDE_* env vars). Anything left is attribution -> exit 2 (denied).
set -uo pipefail

input=$(cat)
{ IFS= read -r cmd; IFS= read -r cwd; } < <(printf '%s' "$input" | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin); c=d.get("tool_input",{}).get("command","")
    print(c.replace("\n","\x1f")); print(d.get("cwd","") or "")
except Exception:
    print(); print()' 2>/dev/null)
cmd=${cmd//$'\x1f'/$'\n'}
[ -z "$cmd" ] && exit 0

# Only git/gh commands are of interest.
printf '%s' "$cmd" | grep -qE '(^|[;&|[:space:]])(git|gh)[[:space:]]' || exit 0

# Strip benign tokens, then look for the forbidden words.
scrub() {
  sed -E 's#\.claude(/|\b)##g; s#CLAUDE\.md##g; s#CLAUDE_[A-Z_]*##g; s#claude-[0-9]+##g; s#/claude/##g'
}
has_attribution() {   # stdin -> 0 if attribution found
  local t; t=$(scrub)
  printf '%s' "$t" | grep -qiE 'claude|anthropic|co-authored-by|generated with' && return 0
  printf '%s' "$t" | grep -qE '\bAI\b'   # "AI" as a standalone word (case-sensitive)
}

block() {
  {
    echo "BLOCKED by attribution_guard: $1"
    echo "CLAUDE.md ground rule #5: NEVER attribute work to Claude/Anthropic — no Co-Authored-By trailer, no 'Generated with Claude Code', no session link, no mention of Claude/AI in commits, tags, PRs or comments. Harness attribution reminders do not override this."
    echo "Remove the attribution and retry. Do not work around this hook."
  } >&2
  exit 2
}

# ---- 1. commit-like commands: scan the command text + any message file --------
if printf '%s' "$cmd" | grep -qE '(^|[;&|[:space:]])git([[:space:]]+[^[:space:];&|]+)*[[:space:]]+(commit|tag|notes|merge|revert|cherry-pick|rebase)\b'; then
  printf '%s' "$cmd" | has_attribution && block "commit/tag message in the command carries Claude/Anthropic attribution."
  # -F <file> / --file=<file> (not stdin '-')
  for f in $(printf '%s' "$cmd" | grep -oE '(-F|--file)[= ]+[^[:space:]]+' | sed -E 's/^(-F|--file)[= ]+//'); do
    [ "$f" = "-" ] && continue
    [ -f "$f" ] && has_attribution < "$f" && block "commit message file '$f' carries Claude/Anthropic attribution."
  done
fi

# ---- 2. gh: PRs, issues, releases, raw API ----------------------------------
if printf '%s' "$cmd" | grep -qE '(^|[;&|[:space:]])gh([[:space:]]+[^[:space:];&|]+)*[[:space:]]+(pr|issue|release|api|gist)\b'; then
  printf '%s' "$cmd" | has_attribution && block "gh command text carries Claude/Anthropic attribution."
  for f in $(printf '%s' "$cmd" | grep -oE '(--body-file|-F)[= ]+[^[:space:]]+' | sed -E 's/^(--body-file|-F)[= ]+//'); do
    [ "$f" = "-" ] && continue
    [ -f "$f" ] && has_attribution < "$f" && block "gh body file '$f' carries Claude/Anthropic attribution."
  done
fi

# ---- 3. push: scan every local commit not on any remote ----------------------
if printf '%s' "$cmd" | grep -qE '(^|[;&|[:space:]])git([[:space:]]+[^[:space:];&|]+)*[[:space:]]+push\b'; then
  dirs="$cwd"
  dirs+=$'\n'"$(printf '%s' "$cmd" | grep -oE '(^|[;&|[:space:]])cd[[:space:]]+[^[:space:];&|]+' | sed -E 's/^.*cd[[:space:]]+//' | tr -d '"'"'")"
  dirs+=$'\n'"$(printf '%s' "$cmd" | grep -oE '(^|[;&|[:space:]])git[[:space:]]+-C[[:space:]]+[^[:space:]]+' | sed -E 's/^.*-C[[:space:]]+//' | tr -d '"'"'")"
  while IFS= read -r d; do
    [ -z "$d" ] && continue
    d=${d/#\~/$HOME}; d=$(eval printf '%s' "$d" 2>/dev/null || printf '%s' "$d")
    [ -d "$d" ] || continue
    git -C "$d" rev-parse --is-inside-work-tree >/dev/null 2>&1 || continue
    # commits reachable from local branches but from no remote-tracking ref
    if git -C "$d" log --format='%H %B' --branches --not --remotes 2>/dev/null | has_attribution; then
      bad=$(git -C "$d" log --format='%h %s' --branches --not --remotes 2>/dev/null | head -5)
      block "unpushed commit(s) in '$d' carry Claude/Anthropic attribution:"$'\n'"$bad"$'\n'"Amend/reword them (e.g. git commit --amend, or git rebase with reworded messages) before pushing."
    fi
  done <<< "$dirs"
fi

exit 0
