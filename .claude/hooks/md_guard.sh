#!/usr/bin/env bash
# PreToolUse(Write) hook — block creation of NEW scattered .md/report files.
#
# Why: CLAUDE.md ground rule #3 says all guidance lives in the one central
# CLAUDE.md — no scattered .md / memory / findings / report dumps. This makes the
# rule non-negotiable at the moment of creation instead of relying on the model
# to not invent exceptions ("it's a report, not guidance" is NOT an exception).
#
# Scope: only Write (create/overwrite) of a .md that does NOT already exist.
# Editing an existing .md is always fine. Baked exemptions: CLAUDE.md, README*,
# anything under .claude/, and the scratchpad/scratch dirs.
#
# Escape hatch (recorded): add the approved new .md's repo-relative or absolute
# path to .claude/md_allowlist.txt (one per line).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ALLOWLIST="$REPO/.claude/md_allowlist.txt"

input=$(cat)
fp=$(printf '%s' "$input" | python3 -c 'import sys,json;
try:
    print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception:
    print("")' 2>/dev/null)
[ -z "$fp" ] && exit 0

# Only guard markdown.
case "$fp" in
  *.md|*.markdown) ;;
  *) exit 0 ;;
esac

# Baked exemptions (meta / scratch).
case "$fp" in
  */CLAUDE.md|*/README.md|*/README*.md|*/.claude/*|*/scratchpad/*|*/scratch/*) exit 0 ;;
esac

# Editing an existing file is fine — only NEW files are "scattering".
[ -e "$fp" ] && exit 0

# User-approved new docs: allowlist match (exact path, repo-relative or absolute).
rel=${fp#"$REPO"/}
if [ -f "$ALLOWLIST" ]; then
  while IFS= read -r line; do
    line=$(printf '%s' "$line" | sed 's/#.*//; s/^[[:space:]]*//; s/[[:space:]]*$//')
    [ -z "$line" ] && continue
    if [ "$line" = "$fp" ] || [ "$line" = "$rel" ]; then exit 0; fi
  done < "$ALLOWLIST"
fi

# Block. exit 2 => tool call denied, stderr shown to the model.
{
  echo "BLOCKED by md_guard: refusing to create new markdown file '$rel'."
  echo "CLAUDE.md ground rule #3: no scattered .md / findings / report files — all guidance goes in the central CLAUDE.md. 'It's a report, not guidance' is NOT an exception."
  echo "If the user has explicitly approved THIS file, record it: add its path to .claude/md_allowlist.txt, then retry. Otherwise, ask the user first."
} >&2
exit 2
