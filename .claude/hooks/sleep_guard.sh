#!/usr/bin/env bash
# PreToolUse(Bash) hook — HARD BLOCK on foreground waiting: `sleep`, `watch`,
# `tail -f`/`-F`, in a command that is not run in the background.
#
# Why: CLAUDE.md (Job submission) — waiting is done by `tools/wait_runs.py` /
# `tools/sweep_waves.py` as background tasks, which notify the session when they
# exit; untracked run IDs are listed by the job_tracking_guard Stop hook. A
# foreground `sleep N; tail log` only blocks the user. The model did it anyway
# (sleep 100/110/150 while the user waited for an answer), so it is enforced here
# instead of trusted.
#
# Allowed: the same commands with run_in_background (e.g. `sleep 240; python3
# tools/wait_runs.py --pending`), and `sleep` inside Python/heredoc code
# (time.sleep) since only shell command positions are matched.
set -uo pipefail

input=$(cat)
{ IFS= read -r bg; IFS= read -r cmd; } < <(printf '%s' "$input" | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin); t=d.get("tool_input",{})
    print("1" if t.get("run_in_background") else "0")
    print(t.get("command","").replace("\n","\x1f"))
except Exception:
    print("0"); print()' 2>/dev/null)
[ "$bg" = "1" ] && exit 0
cmd=${cmd//$'\x1f'/$'\n'}
[ -z "$cmd" ] && exit 0

sep='(^|[;&|(`]|\$\(|[[:space:]](do|then|else)[[:space:]])[[:space:]]*'
if printf '%s\n' "$cmd" | grep -qE "${sep}(sleep|watch)[[:space:]]" \
   || printf '%s\n' "$cmd" | grep -qE "${sep}tail[[:space:]]([^;&|]*[[:space:]])?-[a-zA-Z]*[fF]"; then
  cat >&2 <<'EOF'
BLOCKED by .claude/hooks/sleep_guard.sh: foreground waiting (sleep / watch / tail -f).
Background tools notify you when they exit; the Stop hook lists untracked run IDs.
Do not wait: answer the user / do the next thing now. If something must be waited
on, use tools/wait_runs.py (or the tool itself) with run_in_background.
EOF
  exit 2
fi
exit 0
