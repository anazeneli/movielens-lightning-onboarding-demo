# prep_train_pipeline.py
#
# The minimal demo pipeline -- two ordered steps, nothing else:
#
#     data-prep (CPU)  ->  train (CPU by default)
#
# data-prep runs the real one-time setup scripts (fetch_data.py +
# optimize_data.py) -- both idempotent, so a re-run on already-prepped data
# prints "nothing to do" and moves on. train then runs train_movielens.py
# against the LitData copy data-prep left behind.
#
# The steps run on separate machines and share no local disk, so the data must
# live on the teamspace shared drive (/teamspace/lightning_storage/data/...,
# the recsys.constants defaults). That is the hand-off between the two steps.
# The model then leaves the pipeline via the registry (log_model=True), not disk.
#
# For the full prep -> train -> eval -> serve lifecycle, see lifecycle_pipeline.py.

import argparse
import pathlib
from datetime import datetime

from lightning_sdk import Machine, Studio
from lightning_sdk.pipeline import JobStep, Pipeline

from recsys.constants import LITDATA_DIR, RAW_DATA_DIR

PROJECT = "ml-100k"
WORKFLOW = "train_movielens"

# Jobs run with cwd = studio root, not this repo -- absolute paths only.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Launch the data-prep -> train pipeline.")
parser.add_argument("--train_machine", default="CPU",
                    help="Machine for the training step. Default CPU -- this dataset "
                         "gains nothing from a GPU.")
parser.add_argument("--max_epochs", type=int, default=5, help="Epochs for the training step.")
parser.add_argument("--smoke_test", action="store_true",
                    help="Train for 1 epoch / 2 batches -- verifies the pipeline end to end.")
args = parser.parse_args()

studio = Studio()
teamspace = studio.teamspace
run_id = f"{datetime.now():%Y%m%d-%H%M%S}"

# Flat, slash-free: log_model registers the checkpoint under this name.
experiment_name = f"{PROJECT}-prep-train-{run_id}"

# Pin the data paths into both steps so they agree on the hand-off location,
# whatever this Studio's MOVIELENS_* env vars were when the pipeline was built.
data_env = {"MOVIELENS_DATA_DIR": RAW_DATA_DIR, "MOVIELENS_LITDATA_DIR": LITDATA_DIR}

train_cmd = (
    f"python {REPO_ROOT}/training/train_movielens.py "
    f"--max_epochs {args.max_epochs} "
    f"--logger_name {experiment_name} "
    f"--project {PROJECT} --workflow {WORKFLOW} "
    f"--experiment_group pipeline-{run_id} --experiment_name {experiment_name} "
    f"--sweep_id {run_id}"
    + (" --smoke_test" if args.smoke_test else "")
)

steps = [
    JobStep(
        name="data-prep",
        machine=Machine.CPU,
        env=data_env,
        command=(
            f"cd {REPO_ROOT} && "
            f"python training/fetch_data.py && "
            f"python training/optimize_data.py"
        ),
    ),
    # wait_for defaults to a linear dependency on the previous step.
    JobStep(
        name="train",
        machine=Machine.from_str(args.train_machine),
        env=data_env,
        command=train_cmd,
    ),
]

pipeline_name = f"{PROJECT}-prep-train-{run_id}"
# shared_filesystem=False: required on baremetal (see lifecycle_pipeline.py);
# the hand-off is the teamspace drive, not the pipeline's shared filesystem.
pipeline = Pipeline(name=pipeline_name, studio=studio, shared_filesystem=False)
pipeline.run(steps=steps)

print(f"Launched pipeline '{pipeline_name}'")
print(f"  data-prep (CPU) -> train ({args.train_machine})")
print(f"  data            : {RAW_DATA_DIR} -> {LITDATA_DIR}")
print(f"  experiment name : {experiment_name}")
print(f"\nStep logs: lightning pipeline logs {pipeline_name} <step> --teamspace "
      f"{teamspace.owner.name}/{teamspace.name}")
