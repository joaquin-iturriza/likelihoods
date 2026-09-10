#!/usr/bin/env bash
# Regenerate the public `main` branch from the `lxplus` dev trunk, containing ONLY
# the paths listed in .claude/public_paths.txt. `main` is a BUILD ARTIFACT — never
# edit it by hand, never merge lxplus->main. To change what's public, edit the
# allowlist and re-run this.
#
# Builds an ORPHAN single commit (clean, no dev history) in a throwaway worktree,
# so the main checkout is never disturbed.
#
# Usage:  scripts/publish_main.sh [--no-push]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="lxplus"
ALLOW="$REPO/.claude/public_paths.txt"
cd "$REPO"

PUSH=1
[ "${1:-}" = "--no-push" ] && PUSH=0

git rev-parse --verify "$SRC" >/dev/null 2>&1 || { echo "ERROR: no '$SRC' branch" >&2; exit 1; }
[ -f "$ALLOW" ] || { echo "ERROR: no allowlist $ALLOW" >&2; exit 1; }

git branch -D _pubtmp 2>/dev/null || true
WT="$(mktemp -d)"
# The repo may sit on an sshfs mount owned by a foreign uid; a throwaway worktree
# whose gitdir points into it trips git's "dubious ownership" check. Whitelist it
# for this process only.
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$WT"
cleanup() {
  cd "$REPO" 2>/dev/null || true
  git worktree remove --force "$WT" 2>/dev/null || true
  git branch -D _pubtmp 2>/dev/null || true
}
trap cleanup EXIT

git worktree add --quiet --detach "$WT" "$SRC"
cd "$WT"
git checkout --quiet --orphan _pubtmp
git reset --quiet          # empty the index; working tree (= lxplus tree) intact

staged=0
while IFS= read -r line; do
  line="$(printf '%s' "$line" | sed 's/#.*//; s/^[[:space:]]*//; s/[[:space:]]*$//')"
  [ -z "$line" ] && continue
  if [ -e "$line" ]; then
    git add -f "$line" && staged=$((staged + 1))
  else
    echo "  (skip missing: $line)"
  fi
done < "$ALLOW"
[ "$staged" -gt 0 ] || { echo "ERROR: nothing staged from allowlist" >&2; exit 1; }

git commit --quiet -m "Public core (generated from $SRC by publish_main.sh)"
NEWSHA="$(git rev-parse HEAD)"
cd "$REPO"
git branch -f main "$NEWSHA"
echo "Rebuilt main @ ${NEWSHA:0:10} with $staged allowlisted entries."

if [ "$PUSH" -eq 1 ]; then
  git push --force-with-lease origin main && echo "Pushed main to origin."
else
  echo "(--no-push) main not pushed. Review with: git log --stat main -1"
fi
