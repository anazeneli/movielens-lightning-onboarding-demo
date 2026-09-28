# sweep_launcher.py
#
# Launches a lr x batch_size grid as separate Lightning jobs. Each job is its
# own experiment (litlogger has no cross-experiment "version" concept -- see
# training/README.md, "Grouping experiments").
#
# Naming: each sweep lands in its own folder in the experiment manager --
# litlogger treats every "/" in the experiment name as a folder level, so runs
# appear as "{project}/{sweep_id}/{run}". The model registry can't take those
# extra "/" (it uses "/" only as the owner/teamspace/model_name delimiter), so
# train_movielens.py registers the checkpoint under the flat --logger_name via
# checkpoint_name=, and only the *experiment* gets the folder path
# (--experiment_folder). logger_name stays "{project}-{sweep_id}-lr{lr}-bs{bs}":
# unique across sweeps, and what serving resolves (EXPERIMENT_NAME ->
# "{owner}/{teamspace}/{logger_name}").

import argparse
import pathlib
from datetime import datetime

from lightning_sdk import Studio, Job, Machine

PROJECT = "ml-100k"
WORKFLOW = "train_movielens"

# One machine for both the smoke test and the grid, deliberately. The smoke test
# exists to rule out "it passed there and failed here", so it has to run on the
# same hardware as the real sweep -- GPU image, driver and --precision 16
# behaviour are all generation-specific. A single default, overridable in one
# place, means the two can't drift apart.
MACHINE = Machine.T4

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke_test", action="store_true",
    help="Launch a single remote job running train_movielens.py --smoke_test "
         "instead of the full grid -- verifies the remote job path (absolute "
         "repo path, recsys install, litlogger) before spending on the real sweep. "
         "Runs on the same machine as the real sweep, so it actually proves that path.",
)
parser.add_argument(
    "--machine", default=None,
    help=f"Override the machine for this launch (default: {MACHINE}). Applies to "
         f"the smoke test and the grid alike, so they always match. e.g. T4, L4, "
         f"A100, H100, H100_X_8. The full grid is 6 jobs, so the machine you "
         f"pick here is billed 6 times over -- T4 shows the same parallelism far "
         f"more cheaply than H100.",
)
args = parser.parse_args()

# Resolve once and echo it, so what's about to be billed is never a surprise --
# especially with --machine, where the default in the file no longer tells you
# what's running.
machine = Machine.from_str(args.machine) if args.machine else MACHINE
source = "--machine" if args.machine else "default MACHINE"
print(f"Machine: {machine}  (from {source})")

studio = Studio()

# Jobs run with cwd = studio root, not this repo -- use an absolute path.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

if args.smoke_test:
    timestamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    job_name = f"sweep-launcher-smoke-test-{timestamp}"
    sweep_id = timestamp
    experiment_group = sweep_id
    # Flat registry name; the experiment itself goes in the sweep's folder.
    experiment_name = f"{PROJECT}-{sweep_id}-smoke-test"
    logger_name = experiment_name
    experiment_folder = f"{PROJECT}/{sweep_id}"
    cmd = (
        f"python {REPO_ROOT}/training/train_movielens.py --smoke_test "
        f"--logger_name {logger_name} --experiment_folder {experiment_folder} "
        f"--project {PROJECT} --workflow {WORKFLOW} "
        f"--experiment_group {experiment_group} --experiment_name {experiment_name} "
        f"--sweep_id {sweep_id}"
    )
    Job.run(name=job_name, machine=machine, studio=studio, command=cmd)
    print(f"Launched {job_name} → `{cmd}`")
    print(f"\nCheck this job's logs in the Jobs UI to confirm the remote path works "
          f"end to end, then rerun without --smoke_test for the real sweep.")
    raise SystemExit

# Wider than the original ml100k-sweep grid: pushes past the range where lr
# clearly hurts convergence on either end, so the sweep shows the tradeoff
# curve instead of three similar-looking runs.
learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes    = [128, 256]
sweep_id = f"{datetime.now():%Y%m%d-%H%M%S}"
experiment_group = sweep_id

grid = [(lr, bs) for lr in learning_rates for bs in batch_sizes]
print(f"Launching {len(grid)} job(s) on {machine} -- each is billed separately.\n")

for idx, (lr, bs) in enumerate(grid):
    # Flat registry name (see header); lr + bs are the only params this grid
    # varies, so the full string is unique. The experiment goes in the
    # {project}/{sweep_id} folder, so one sweep's runs sit together in the UI.
    experiment_name = f"{PROJECT}-{sweep_id}-lr{lr}-bs{bs}"
    logger_name = experiment_name
    experiment_folder = f"{PROJECT}/{sweep_id}"
    # Job.run's own name -- unrelated to litlogger, just the Jobs UI label.
    job_name = f"sweep-{experiment_name}"

    cmd      = (
        f"python {REPO_ROOT}/training/train_movielens.py "
        f"--lr {lr} "
        f"--batch_size {bs} "
        f"--precision 16 "
        f"--max_epochs 25 "
        f"--logger_name {logger_name} --experiment_folder {experiment_folder} "
        f"--project {PROJECT} --workflow {WORKFLOW} "
        f"--experiment_group {experiment_group} --experiment_name {experiment_name} "
        f"--sweep_id {sweep_id}"
    )

    # NOTE: lightning_sdk's Job.run() replaces the old
    # Studio.install_plugin('jobs') API, which no longer exists in this
    # SDK version -- machine= is now required, there's no implicit
    # "current machine" default.
    Job.run(name=job_name, machine=machine, studio=studio, command=cmd)

    print(f"Launched {job_name} → `{cmd}`")

print(f"\nAll {len(grid)} runs are in the '{PROJECT}/{sweep_id}' folder in the experiment "
      f"manager -- compare them there, pick the best config, then run that config's "
      f"full training with its own --logger_name.")