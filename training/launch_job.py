# training/launch_job.py
#
# Launch train_movielens.py as a single remote job with arbitrary
# hyperparameters -- e.g. a longer run on a sweep's winning config (see
# README.md, "Smoke test locally, then experiment", step 4). Not part of a
# sweep: no --experiment_group is set, so this lands as its own flat
# experiment, not inside the sweep's folder.
#
# Anchored to this file's own location (not the caller's cwd) so it can be
# run from anywhere -- see training/README.md, "Grouping experiments" for why
# that matters (jobs run with cwd = studio root, not this repo).
#
# Example:
#     python training/launch_job.py --lr 0.001 --batch_size 128 \
#         --max_epochs 100 --logger_name ml100k-best

import argparse
import pathlib

from lightning_sdk import Studio, Job, Machine

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--precision", type=int, default=16, choices=[16, 32])
parser.add_argument("--logger_name", type=str, required=True)
parser.add_argument("--machine", type=str, default="H100", help="Any lightning_sdk.Machine name, e.g. H100, CPU, A100")
parser.add_argument("--job_name", type=str, default=None)
args = parser.parse_args()

studio = Studio()
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

cmd = (
    f"python {REPO_ROOT}/training/train_movielens.py "
    f"--lr {args.lr} --batch_size {args.batch_size} --precision {args.precision} "
    f"--max_epochs {args.max_epochs} --logger_name {args.logger_name}"
)
job_name = args.job_name or f"{args.logger_name}-{args.max_epochs}ep"
machine = getattr(Machine, args.machine.upper())

Job.run(name=job_name, machine=machine, studio=studio, command=cmd)
print(f"Launched {job_name} → `{cmd}`")
