#!/usr/bin/env bash
# Stop hook — push any unpushed commits to origin automatically at end of turn.
#
# Why: the git workflow says "push after every significant change"; relying on the
# model to remember (and not ask permission) is unreliable. This makes it
# deterministic. It only pushes work that is ALREADY COMMITTED — it never creates
# commits, and it NEVER pushes `main` (a generated build artifact; publish it with
# scripts/publish_main.sh instead). Failures are non-fatal (printed, exit 0) so a
# push problem never blocks the session.
set -uo pipefail
REPO="/eos/home-j/joiturri/likelihoods"
cd "$REPO" 2>/dev/null || exit 0

# Walk every worktree (the main checkout + any ../wt-* feature worktrees) so
# feature-branch commits get pushed too.
git worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2}' | while read -r wt; do
  br=$(git -C "$wt" symbolic-ref --quiet --short HEAD 2>/dev/null) || continue
  [ -z "$br" ] && continue
  [ "$br" = "main" ] && continue   # main is a generated artifact — published via scripts/publish_main.sh, never auto-pushed

  if git -C "$wt" rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    ahead=$(git -C "$wt" rev-list --count '@{u}'..HEAD 2>/dev/null || echo 0)
    if [ "${ahead:-0}" -gt 0 ]; then
      if git -C "$wt" push --quiet 2>/dev/null; then
        echo "[auto-push] $br: pushed $ahead commit(s) to origin"
      else
        echo "[auto-push] $br: push of $ahead commit(s) FAILED (push manually)"
      fi
    fi
  else
    # New branch with no upstream yet — publish it.
    if git -C "$wt" push --quiet -u origin "$br" 2>/dev/null; then
      echo "[auto-push] $br: published new branch to origin"
    else
      echo "[auto-push] $br: publish FAILED (push manually)"
    fi
  fi
done
exit 0
