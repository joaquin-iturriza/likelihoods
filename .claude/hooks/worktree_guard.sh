#!/usr/bin/env bash
# PreToolUse(Edit|Write|MultiEdit) hook — worktree reminder.
#
# Why: "create a worktree for new feature work" has no natural trigger, so the
# model forgets it. This supplies the trigger at the moment it matters: the first
# edit of trunk code on `main`. ADVISORY (non-blocking) — it injects a reminder
# and lets the edit proceed, so false positives cost one line, not a hard stop.
set -uo pipefail
REPO="/eos/home-j/joiturri/likelihoods"

input=$(cat)
fp=$(printf '%s' "$input" | python3 -c 'import sys,json;
try:
    print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception:
    print("")' 2>/dev/null)
[ -z "$fp" ] && exit 0

# Only the MAIN checkout matters; feature worktrees live outside REPO (../wt-*).
case "$fp" in
  "$REPO"/*) ;;
  *) exit 0 ;;
esac

# Exempt meta / lightweight files that don't warrant a feature worktree.
case "$fp" in
  */.claude/*|*/CLAUDE.md|*.md|*/scratchpad/*|*/scratch/*) exit 0 ;;
esac

# Only nudge when actually sitting on the trunk branch in the main checkout.
br=$(git -C "$REPO" symbolic-ref --quiet --short HEAD 2>/dev/null)
[ "$br" = "lxplus" ] || exit 0

rel=${fp#"$REPO"/}
msg="Worktree reminder: editing trunk file '$rel' directly on lxplus. Per the git workflow, non-trivial feature work should go in a worktree (git worktree add ../wt-<feat> -b <feat> lxplus). If this is a quick standalone edit, proceed; otherwise create a worktree first."
printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":%s}}\n' \
  "$(printf '%s' "$msg" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')"
exit 0
