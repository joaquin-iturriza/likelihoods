#!/usr/bin/env bash
# wait_for_jobs.sh — block until a set of HTCondor jobs leave the queue.
#
# Call this after every condor_submit so submitted work is actually tracked to
# completion (see CLAUDE.md — never fire-and-forget). Intended to be launched in
# the background; it prints a progress line each poll and exits when done.
#
# Usage:
#   scripts/wait_for_jobs.sh <clusterId> [<clusterId> ...]   # wait on specific clusters
#   scripts/wait_for_jobs.sh --mine                          # wait on all of $USER's jobs
#   scripts/wait_for_jobs.sh --constraint 'EXPR'             # condor ClassAd constraint
#   scripts/wait_for_jobs.sh --sweep-dir <AFS sweep dir>     # jobs whose logs are in <dir>/log
#
# Env:
#   POLL_INTERVAL   seconds between polls (default 120)
#   HELD_GRACE      consecutive all-held polls before giving up (default 5)
#
# Exit: 0 = all jobs finished (completed/removed); 2 = usage; 3 = jobs stuck Held.
set -uo pipefail

POLL_INTERVAL="${POLL_INTERVAL:-120}"
HELD_GRACE="${HELD_GRACE:-5}"

usage() {
    echo "usage: $0 <clusterId...> | --mine | --constraint 'EXPR' | --sweep-dir DIR" >&2
    exit 2
}
[ $# -ge 1 ] || usage

owner="${USER:-$(id -un)}"
QARGS=()  # extra args appended to condor_q to select the jobs of interest
case "$1" in
    --mine)
        QARGS=("$owner") ;;
    --constraint)
        [ -n "${2:-}" ] || usage
        QARGS=(-constraint "$2") ;;
    --sweep-dir)
        dir="${2:-}"; [ -n "$dir" ] || usage
        # cluster ids from the per-job logs condor writes: <name>.<cluster>.<proc>.log
        mapfile -t ids < <(ls "$dir"/log/*.log 2>/dev/null \
            | sed -E 's/.*\.([0-9]+)\.[0-9]+\.log$/\1/' | sort -un)
        [ ${#ids[@]} -gt 0 ] || { echo "no job logs under $dir/log yet" >&2; exit 2; }
        QARGS=("${ids[@]}") ;;
    -*)
        usage ;;
    *)
        QARGS=("$@") ;;  # explicit cluster ids
esac

echo "[wait_for_jobs] tracking: ${QARGS[*]}  (poll every ${POLL_INTERVAL}s)"
held_streak=0
while true; do
    if ! out=$(condor_q "${QARGS[@]}" -af JobStatus 2>/dev/null); then
        echo "[wait_for_jobs] condor_q unavailable, retrying in ${POLL_INTERVAL}s..." >&2
        sleep "$POLL_INTERVAL"; continue
    fi
    n=$(grep -c '[0-9]' <<<"$out")
    if [ "$n" -eq 0 ]; then
        echo "[wait_for_jobs] $(date '+%F %T') — all jobs finished."
        exit 0
    fi
    idle=$(grep -cx '1' <<<"$out"); run=$(grep -cx '2' <<<"$out"); held=$(grep -cx '5' <<<"$out")
    printf '[wait_for_jobs] %s  in-queue=%d  idle=%d running=%d held=%d\n' \
        "$(date '+%T')" "$n" "$idle" "$run" "$held"
    if [ "$held" -eq "$n" ]; then
        held_streak=$((held_streak + 1))
        if [ "$held_streak" -ge "$HELD_GRACE" ]; then
            echo "[wait_for_jobs] $held job(s) stuck Held — aborting wait." >&2
            exit 3
        fi
    else
        held_streak=0
    fi
    sleep "$POLL_INTERVAL"
done
