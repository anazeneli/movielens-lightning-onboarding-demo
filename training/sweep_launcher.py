# sweep_launcher.py
#
# Launches a lr x batch_size grid as separate Lightning jobs, all logging into
# ONE experiment in the experiment manager so they can be compared side by side
# (rather than 9 unrelated experiments). See training/README.md, "Grouping
# experiments" for how this works.

import argparse
import pathlib

from lightning_sdk import Studio, Job, Machine

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
    job_name = "sweep-launcher-smoke-test"
    cmd = f"python {REPO_ROOT}/training/train_movielens.py --smoke_test"
    Job.run(name=job_name, machine=Machine.CPU, studio=studio, command=cmd)
    print(f"Launched {job_name} → `{cmd}`")
    print(f"\nCheck this job's logs in the Jobs UI to confirm the remote path works "
          f"end to end, then rerun without --smoke_test for the real sweep.")
    raise SystemExit

# Original sweep, already logged under
# 'ml100k-sweep' in the experiment manager. The active grid below is a
# separate experiment (new name + wider lr range), not a rerun of this one.
# learning_rates = [1e-4, 1e-3, 1e-2]
# batch_sizes    = [128, 256, 512]
# EXPERIMENT_GROUP = "ml100k-sweep"

# Wider than the original grid: pushes past the range where lr clearly hurts
# convergence on either end, so the experiment manager shows the tradeoff
# curve instead of three similar-looking runs.
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
batch_sizes    = [128, 256, 512]
EXPERIMENT_GROUP = "ml100k-sweep-wide-lr"

grid = [(lr, bs) for lr in learning_rates for bs in batch_sizes]

# Shared across every job in this sweep: same --logger_name means all runs
# become versions of ONE experiment, not N separate experiments. lr/batch_size
# are still logged as metadata per run (see train_movielens.py), so each
# version is distinguishable when comparing results.

for idx, (lr, bs) in enumerate(grid):
    # Unique per-job name -- only used to tell jobs apart in the Jobs UI.
    job_name = f"{EXPERIMENT_GROUP}-{idx}-lr{lr}-bs{bs}"

    cmd      = (
        f"python {REPO_ROOT}/training/train_movielens.py "
        f"--lr {lr} "
        f"--batch_size {bs} "
        f"--precision 16 "
        f"--max_epochs 25 "
        f"--logger_name {EXPERIMENT_GROUP}"
    )

    # Machine.CPU keeps this sweep free -- swap for Machine.H100 (or any other
    # Machine option) once you're past a dry run and want the real timings.
    # NOTE: lightning_sdk's Job.run() replaces the old
    # Studio.install_plugin('jobs') API, which no longer exists in this
    # SDK version -- machine= is now required, there's no implicit
    # "current machine" default.
    Job.run(name=job_name, machine=Machine.CPU, studio=studio, command=cmd)

    print(f"Launched {job_name} → `{cmd}`")

print(f"\nAll {len(grid)} runs grouped under experiment '{EXPERIMENT_GROUP}' -- compare their "
      f"versions in the experiment manager to pick the best config, then run that "
      f"config's full training with its own --logger_name.")