# training/smoke_test_job.py
#
# Remote counterpart to `train_movielens.py --smoke_test`. That one proves the
# code works in THIS Studio; this one proves it works as a Lightning Job, which
# is a different environment: a fresh machine, cwd = studio root rather than
# this repo, and the teamspace drive reached over the network instead of a local
# mount. Most setup breakages (an absolute path that only exists here, data left
# on local disk, a missing package) only show up on that side.
#
# Unlike `sweep_launcher.py --smoke_test`, which launches and returns, this
# waits for the job to finish and checks its logs for the success marker, so it
# exits non-zero when the remote path is broken.
#
#     python training/smoke_test_job.py                  # CPU, waits up to 20 min
#     python training/smoke_test_job.py --machine L4     # same, on a GPU
#     python training/smoke_test_job.py --no_wait        # launch only, don't wait
#
# Billed like any job: one short CPU run, a few minutes.

import argparse
import pathlib
import sys
import time
from datetime import datetime

from lightning_sdk import Job, Machine, Studio

SUCCESS_MARKER = "Smoke test passed"
TERMINAL = {"completed", "succeeded", "failed", "stopped", "cancelled", "crashed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", default="CPU",
                        help="Any lightning_sdk.Machine name (default: CPU -- the cheapest "
                             "thing that proves the path works)")
    parser.add_argument("--timeout", type=int, default=1200,
                        help="Seconds to wait for the job before giving up (default: 1200)")
    parser.add_argument("--no_wait", action="store_true",
                        help="Launch and exit without waiting. Check the Jobs UI yourself.")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the job after a pass (default: delete it, so repeated "
                             "smoke tests don't pile up in the Jobs UI)")
    args = parser.parse_args()

    studio = Studio()
    # Jobs run with cwd = studio root, not this repo -- absolute path required.
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    # Unique logger name per run: the registry rejects re-registering a
    # (name, version) pair, and a fixed name would collide on the second run.
    logger_name = f"job-smoke-test-{stamp}"
    command = (
        f"python {repo_root}/training/train_movielens.py --smoke_test "
        f"--logger_name {logger_name}"
    )

    machine = Machine.from_str(args.machine) if isinstance(args.machine, str) else args.machine
    print(f"Launching job on {machine}\n  {command}")
    job = Job.run(name=f"job-smoke-test-{stamp}", machine=machine, studio=studio, command=command)
    print(f"Job: {job.name}\n  {job.link}")

    if args.no_wait:
        print("--no_wait given; not waiting. Check the Jobs UI for the result.")
        return 0

    deadline = time.time() + args.timeout
    last = None
    while time.time() < deadline:
        status = str(job.status).lower().rsplit(".", 1)[-1]
        if status != last:
            print(f"  status: {status}")
            last = status
        if status in TERMINAL:
            break
        time.sleep(10)
    else:
        print(f"FAIL: job still {last} after {args.timeout}s. It may just be slow to "
              f"schedule -- check {job.link} before assuming the code is broken.")
        return 1

    # Status alone isn't enough: a training script that died with a traceback
    # has still been observed to report Completed, so check for the marker the
    # smoke test prints on success.
    logs = job.logs or ""
    passed = SUCCESS_MARKER in logs
    if passed:
        print(f"PASS: remote job printed {SUCCESS_MARKER!r} (status: {last}).")
        if not args.keep:
            job.delete()
            print("  Job deleted. Pass --keep to retain it.")
        return 0

    print(f"FAIL: job ended {last} without printing {SUCCESS_MARKER!r}.")
    print(f"  Logs: {job.link}")
    tail = [ln for ln in logs.splitlines() if ln.strip()][-30:]
    if tail:
        print("  --- last lines of remote log ---")
        for line in tail:
            print(f"  {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
