"""
Run an initialised DyHPO sweep in waves, so later trials are chosen from earlier results.

    python tools/sweep_waves.py SWEEP --site lxplus --waves 10,5,5 [--first-trial 0]
                                [--running RUN_ID ...] [--mem 8G] [--prio N]

Runs HERE (locally) and reaches the cluster only through `site`. For each wave it
  1. plans with `site pick` (restricted to the sweep's site: the DyHPO state lives
     there, so its trials cannot move) and submits the wave's trials with
     `site submit <site> likelihoods scripts/sweep_trial.sh -- SWEEP <idx>`,
  2. waits until `site poll` reports every job of the wave in a terminal state
     (anything else, connection errors included, counts as still running),
  3. checks in the sweep's DyHPO state that the wave's trials reported a result
     (`site` also calls a removed job COMPLETED), and stops if any did not,
  4. goes on with the next wave.

Why waves: every trial asks the shared DyHPO state for its hyperparameters when it
starts. Submitted all at once, every trial asks before any result exists, which
makes the sweep a random search. The first wave should be the sweep's random
start-up trials (dyhpo.n_startup); each later wave is chosen with all earlier
results in hand.

--prio N (HTCondor sites): set each submitted trial's JobPrio to N with condor_prio.
JobPrio only orders a user's own jobs, so this puts the sweep ahead of the user's
other queued jobs and changes nothing else.

--running RUN_ID ... : a wave that is already submitted (its site run IDs); the tool
waits for it, checks it, then submits the waves after it. Trials are numbered from
--first-trial on, in the order of --waves (the running wave included).

The sweep must have been initialised first:
    site run <site> likelihoods -- python sweep/generate_sweep.py --config <cfg>
"""

import argparse
import re
import subprocess
import sys
import time

TERMINAL = {"COMPLETED", "FAILED", "REMOVED", "CANCELLED", "HELD", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
POLL_SECONDS = 300
# the scheduler refusing a job because the user already has the maximum queued
SUBMIT_CAP = r"QOSMaxSubmitJobPerUserLimit|MaxSubmitJobs|maximum number of jobs"


def site(*args, timeout=600):
    r = subprocess.run(["site", "--timeout", str(timeout), *args], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M] ") + msg, flush=True)


def n_observations(site_name, sweep):
    """Number of trials that reported to the sweep's DyHPO state (results + diverged), or None."""
    # results = successful observations + configs DyHPO recorded as diverged (a trial
    # whose loss went non-finite is a valid sweep outcome, not a missing result)
    code = ("import os, pickle; s = pickle.load(open(os.path.join(os.environ['SUBMIT_DIR'], "
            f"'sweeps', '{sweep}', 'dyhpo_state.pkl'), 'rb')); "
            "print('NOBS', len(s['observations']) + len(s.get('diverged_configs') or []))")
    rc, out = site("run", "--quote", site_name, "likelihoods", "--", "python", "-c", code)
    m = re.search(r"^NOBS (\d+)$", out, re.M)
    return int(m.group(1)) if m else None


def states(runs):
    # read the states from the poll itself ("<run> OLD -> NEW"): `site runs` lists only
    # the most recent runs, so a wave's runs fell off it once other trials were
    # submitted and stayed "?" forever
    rc, out = site("poll", *runs)
    st = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\S+)\s+\S+\s+->\s+(\S+)", line)
        if m and m.group(1) in runs:
            st[m.group(1)] = m.group(2)
    return st


def wait(runs):
    while True:
        st = states(runs)
        done = [r for r in runs if st.get(r) in TERMINAL]
        if len(done) == len(runs):
            return st
        counts = {}
        for r in runs:
            counts[st.get(r, "?")] = counts.get(st.get(r, "?"), 0) + 1
        log(f"waiting: {len(done)}/{len(runs)} finished {counts}")
        time.sleep(POLL_SECONDS)


def submit(site_name, sweep, trials, mem, prio=None):
    rc, out = site("pick", "likelihoods", "--jobs", str(len(trials)), "--groups", "1", "--mem", mem,
                   "--only", site_name)
    log("plan: " + " | ".join(l.strip() for l in out.splitlines() if "done in" in l))
    runs = []
    for k, idx in enumerate(trials):
        args = ["submit", site_name, "likelihoods", "scripts/sweep_trial.sh", "--est-gpu-hours", "1",
                "--note", f"{sweep} trial {idx}"]
        if k:
            args.append("--no-sync")
        while True:
            rc, out = site(*args, "--", sweep, str(idx))
            m = re.search(r"run (\S+)\s+\(", out)
            if (rc or not m) and re.search(SUBMIT_CAP, out):
                log(f"trial {idx}: at {site_name}'s per-user submit cap; retrying in {POLL_SECONDS // 60} min")
                time.sleep(POLL_SECONDS)
                if "--no-sync" not in args:
                    args.append("--no-sync")       # synced on the first attempt
                continue
            break
        if rc or not m:
            sys.exit(f"submitting trial {idx} failed:\n{out}")
        runs.append(m.group(1))
        log(f"trial {idx}: {m.group(1)}")
        if prio is not None:
            job = re.search(r"condor job (\d+)", out)
            if not job:
                sys.exit(f"--prio needs an HTCondor site; no condor job id in:\n{out}")
            rc, pout = site("run", "--quote", site_name, "likelihoods", "--", "condor_prio", "-p", str(prio), job.group(1))
            log(f"trial {idx}: JobPrio {prio}" + ("" if rc == 0 else f" FAILED: {pout.strip()[-200:]}"))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep")
    ap.add_argument("--site", required=True, help="the site holding the sweep's state")
    ap.add_argument("--waves", required=True, help="trials per wave, e.g. 10,5,5")
    ap.add_argument("--first-trial", type=int, default=0)
    ap.add_argument("--running", nargs="*", default=[], help="site run IDs of an already submitted first wave")
    ap.add_argument("--mem", default="8G", help="memory to plan for (the job's own request is the site default)")
    ap.add_argument("--prio", type=int, default=None, help="HTCondor JobPrio for each submitted trial")
    a = ap.parse_args()

    sizes = [int(x) for x in a.waves.split(",")]
    idx = a.first_trial
    n0 = n_observations(a.site, a.sweep)
    if n0 is None:
        sys.exit(f"cannot read the DyHPO state of {a.sweep} on {a.site}: initialise the sweep first")
    log(f"{a.sweep} on {a.site}: {n0} results so far; waves {sizes} from trial {idx}")
    for w, size in enumerate(sizes):
        trials = list(range(idx, idx + size))
        if w == 0 and a.running:
            assert len(a.running) == size, f"--running has {len(a.running)} runs, wave 1 has {size} trials"
            runs = a.running
            log(f"wave 1 (trials {trials[0]}-{trials[-1]}) already submitted")
        else:
            log(f"wave {w + 1}: submitting trials {trials[0]}-{trials[-1]}")
            runs = submit(a.site, a.sweep, trials, a.mem, a.prio)
        st = wait(runs)
        n = n_observations(a.site, a.sweep)
        log(f"wave {w + 1} finished {sorted(set(st.values()))}; results in the state: {n0} -> {n}")
        if n is None or n - n0 < size:
            sys.exit(f"wave {w + 1}: only {None if n is None else n - n0} of {size} trials reported a result; "
                     "read their logs (site logs <run>) before going on")
        n0, idx = n, idx + size
    log(f"all waves done: {n0} results")


if __name__ == "__main__":
    main()
