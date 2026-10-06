#!/bin/bash
# Stop hook: a turn may not end while a submitted job of this project is untracked.
#
# CLAUDE.md: "Always track submitted jobs ... never fire-and-forget." This makes it
# structural: every unfinished likelihoods run in the site registry must be covered
# by a live tools/wait_runs.py started from THIS session (it registers itself in
# ~/.local/share/ccorch/waiters/<pid>.json with the session id). A waiter that is
# detached, or belongs to another session, does not count: it would never notify
# this one. Runs older than 14 days are ignored (stale registry entries).
input=$(cat)
python3 - "$input" <<'EOF'
import glob, json, os, subprocess, sys, time

hook = json.loads(sys.argv[1] or "{}")
session = hook.get("session_id")
TERMINAL = {"COMPLETED", "FAILED", "REMOVED", "CANCELLED", "HELD", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
try:
    out = subprocess.run(["site", "runs", "--project", "likelihoods"], capture_output=True,
                         text=True, timeout=30).stdout
except Exception:
    sys.exit(0)                       # registry unreadable: do not block on it
cutoff = time.time() - 14 * 86400
pending = []
for line in out.splitlines():
    p = line.split()
    if len(p) >= 6 and p[0].startswith("likelihoods-") and p[3] not in TERMINAL:
        try:
            t = time.mktime(time.strptime(p[4] + " " + p[5], "%Y-%m-%d %H:%M"))
        except ValueError:
            t = time.time()
        if t >= cutoff:
            pending.append(p[0])
if not pending:
    sys.exit(0)

covered = set()
for f in glob.glob(os.path.expanduser("~/.local/share/ccorch/waiters/*.json")):
    try:
        w = json.load(open(f))
        os.kill(int(w["pid"]), 0)     # alive?
    except Exception:
        continue
    if session and w.get("session") not in (None, session):
        continue
    covered |= set(w.get("runs", []))

missing = [r for r in pending if r not in covered]
if missing:
    print(json.dumps({"decision": "block", "reason": (
        f"{len(missing)} submitted likelihoods run(s) are not tracked by a waiter of this session: "
        f"{' '.join(missing)}. Start one as a BACKGROUND task (Bash run_in_background) before "
        f"ending the turn:  python tools/wait_runs.py {' '.join(missing)}   "
        "(or: python tools/wait_runs.py --pending). Re-start it when it exits while runs remain.")}))
EOF
exit 0
