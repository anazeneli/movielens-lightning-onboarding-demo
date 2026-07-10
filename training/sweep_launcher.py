# sweep_launcher.py
#
# Launches a lr x batch_size grid as separate Lightning jobs. Each job is its
# own experiment (litlogger has no cross-experiment "version" concept -- see
# training/README.md, "Grouping experiments").
#
# Naming: with log_model=True, litlogger registers the best checkpoint in the
# model registry *under the experiment name*, recombined as
# "{owner}/{teamspace}/{name}". The registry uses "/" only as the
# org/teamspace/model_name delimiter, so the name must be a single flat segment
# with NO "/". An earlier "{project}/{workflow}/{group}/{experiment}" scheme
# gave UI folder hierarchy but made the registered model name unparseable
# (6 slash-parts instead of 3) and blew up the checkpoint upload. Keep it flat
# and short instead: "{project}-{sweep_id}-lr{lr}-bs{bs}". logger_name ==
# experiment_name so the serving side (server.py, EXPERIMENT_NAME ->
# "{owner}/{teamspace}/{experiment_name}") resolves the same string it was
# registered under. project / workflow / group stay as logged metadata, and
# runs from one sweep sort together by their shared "{project}-{sweep_id}-"
# name prefix.

import argparse
import pathlib
from datetime import datetime

from lightning_sdk import Studio, Job, Machine

PROJECT = "ml-100k"
WORKFLOW = "train_movielens"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke_test", action="store_true",
    help="Launch a single remote job running train_movielens.py --smoke_test "
         "instead of the full grid -- verifies the remote job path (absolute "
         "repo path, recsys install, litlogger) before spending on the real sweep.",
)
args = parser.parse_args()

studio = Studio()

# Jobs run with cwd = studio root, not this repo -- use an absolute path.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

if args.smoke_test:
    timestamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    job_name = f"sweep-launcher-smoke-test-{timestamp}"
    sweep_id = timestamp
    experiment_group = sweep_id
    # Flat, slash-free name (see header) -- also what log_model registers under.
    experiment_name = f"{PROJECT}-{sweep_id}-smoke-test"
    logger_name = experiment_name
    cmd = (
        f"python {REPO_ROOT}/training/train_movielens.py --smoke_test "
        f"--logger_name {logger_name} "
        f"--project {PROJECT} --workflow {WORKFLOW} "
        f"--experiment_group {experiment_group} --experiment_name {experiment_name} "
        f"--sweep_id {sweep_id}"
    )
    Job.run(name=job_name, machine=Machine.CPU, studio=studio, command=cmd)
    print(f"Launched {job_name} → `{cmd}`")
    print(f"\nCheck this job's logs in the Jobs UI to confirm the remote path works "
          f"end to end, then rerun without --smoke_test for the real sweep.")
    raise SystemExit

# Wider than the original ml100k-sweep grid: pushes past the range where lr
# clearly hurts convergence on either end, so the sweep shows the tradeoff
# curve instead of three similar-looking runs.
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
batch_sizes    = [128, 256, 512]
sweep_id = f"{datetime.now():%Y%m%d-%H%M%S}"
experiment_group = sweep_id

grid = [(lr, bs) for lr in learning_rates for bs in batch_sizes]

for idx, (lr, bs) in enumerate(grid):
    # Flat, slash-free name (see header) -- also what log_model registers under.
    # {project}-{sweep_id} groups this sweep's runs by shared prefix; lr + bs
    # are the only params this grid varies and every combo is unique, so the
    # full string is a unique experiment_name within the sweep.
    experiment_name = f"{PROJECT}-{sweep_id}-lr{lr}-bs{bs}"
    logger_name = experiment_name
    # Job.run's own name -- unrelated to litlogger, just the Jobs UI label.
    job_name = f"sweep-{experiment_name}"

    cmd      = (
        f"python {REPO_ROOT}/training/train_movielens.py "
        f"--lr {lr} "
        f"--batch_size {bs} "
        f"--precision 16 "
        f"--max_epochs 25 "
        f"--logger_name {logger_name} "
        f"--project {PROJECT} --workflow {WORKFLOW} "
        f"--experiment_group {experiment_group} --experiment_name {experiment_name} "
        f"--sweep_id {sweep_id}"
    )

    # Machine.CPU keeps this sweep free -- swap for Machine.H100 (or any other
    # Machine option) once you're past a dry run and want the real timings.
    # NOTE: lightning_sdk's Job.run() replaces the old
    # Studio.install_plugin('jobs') API, which no longer exists in this
    # SDK version -- machine= is now required, there's no implicit
    # "current machine" default.
    # NOTE: this is set to CPU not H100 for savings during testing
    Job.run(name=job_name, machine=Machine.CPU, studio=studio, command=cmd)

    print(f"Launched {job_name} → `{cmd}`")

print(f"\nAll {len(grid)} runs share the name prefix '{PROJECT}-{experiment_group}-' "
      f"in the experiment manager -- filter/sort by it to compare them, pick the best "
      f"config, then run that config's full training with its own --logger_name.")