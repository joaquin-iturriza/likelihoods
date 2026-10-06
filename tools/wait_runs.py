"""
Wait until site runs reach a terminal state. THE way to track a submission.

    python tools/wait_runs.py RUN_ID [RUN_ID ...]      # as a background task of the session
    python tools/wait_runs.py --pending                 # every unfinished likelihoods run
    python tools/wait_runs.py --once RUN_ID ...         # print the states and exit

Start it as a background task right after a submission (Bash run_in_background),
so the session is notified when it exits. It refreshes the states with `site poll`
every --every seconds and exits when every run is in a terminal state. Anything
else, a connection error written into the registry included, counts as not done.
It cannot finish early on a failed query.

While it runs it registers itself in ~/.local/share/ccorch/waiters/<pid>.json
(the runs it covers and the Claude session that started it). The Stop hook
.claude/hooks/job_tracking_guard.sh blocks ending a turn while an unfinished run
of this project is not covered by a live waiter of the current session.

On exit it prints each run's final state. A terminal state is not success: `site`
also shows a removed job as COMPLETED. Read `site logs <run>` before saying a run
worked.
"""

import argparse
import atexit
import json
import os
import subprocess
import sys
import time

TERMINAL = {"COMPLETED", "FAILED", "REMOVED", "CANCELLED", "HELD", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
PROJECT = "likelihoods"
WAITERS = os.path.expanduser("~/.local/share/ccorch/waiters")


def registry():
    """{run_id: state} for this project, from the local registry."""
    out = subprocess.run(["site", "runs", "--project", PROJECT], capture_output=True, text=True).stdout
    st = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 4 and p[0].startswith(PROJECT + "-"):
            st[p[0]] = p[3]
    return st


def unfinished():
    return sorted(r for r, s in registry().items() if s not in TERMINAL)


def poll(runs):
    subprocess.run(["site", "--timeout", "300", "poll", *runs], capture_output=True, text=True)
    st = registry()
    return {r: st.get(r, "?") for r in runs}


def register(runs):
    os.makedirs(WAITERS, exist_ok=True)
    path = os.path.join(WAITERS, f"{os.getpid()}.json")
    with open(path, "w") as f:
        json.dump({"pid": os.getpid(), "runs": runs, "project": PROJECT,
                   "session": os.environ.get("CLAUDE_CODE_SESSION_ID"),
                   "started": time.strftime("%Y-%m-%d %H:%M")}, f)
    atexit.register(lambda: os.path.exists(path) and os.remove(path))


def summary(st):
    counts = {}
    for s in st.values():
        counts[s] = counts.get(s, 0) + 1
    return " ".join(f"{n} {s}" for s, n in sorted(counts.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--pending", action="store_true", help="wait for every unfinished run of the project")
    ap.add_argument("--once", action="store_true", help="print the states and exit")
    ap.add_argument("--every", type=int, default=300, help="seconds between polls")
    a = ap.parse_args()

    runs = sorted(set(a.runs) | (set(unfinished()) if a.pending else set()))
    if not runs:
        sys.exit("no runs to wait for")
    if a.once:
        st = poll(runs)
        print(f"{len(runs)} runs: {summary(st)}")
        return
    register(runs)
    print(time.strftime("[%H:%M] ") + f"waiting for {len(runs)} runs", flush=True)
    while True:
        st = poll(runs)
        if all(s in TERMINAL for s in st.values()):
            break
        time.sleep(a.every)
    print(time.strftime("[%H:%M] ") + f"all {len(runs)} runs finished: {summary(st)}")
    for r in runs:
        print(f"  {r} {st[r]}")
    print("A terminal state is not success: read `site logs <run>` before reporting.")


if __name__ == "__main__":
    main()
