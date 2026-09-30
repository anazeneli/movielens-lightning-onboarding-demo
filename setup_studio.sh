#!/usr/bin/env bash
# setup_studio.sh
#
# One-shot setup for a fresh Studio: installs the package, makes sure the
# teamspace data folder exists and is mounted, fetches + optimizes MovieLens,
# then runs the smoke test. Every step is idempotent, so re-running is safe.
#
# The remote-Job smoke test is opt-in (--with-job) because it bills compute.
#
#     bash setup_studio.sh               # full setup
#     bash setup_studio.sh --skip-smoke  # everything except the smoke test
#     bash setup_studio.sh --with-job    # also smoke test the remote Job path
#
# The data folder step is the one fetch_data.py can't do for itself:
# /teamspace/lightning_storage/ is not writable, only its mounted subfolders
# are, so the folder has to be created through the SDK (see README.md).

set -euo pipefail
cd "$(dirname "$0")"

SKIP_SMOKE=0
WITH_JOB=0
for arg in "$@"; do
    case "$arg" in
        --skip-smoke) SKIP_SMOKE=1 ;;
        --with-job)   WITH_JOB=1 ;;
        *) echo "unknown option: $arg (expected --skip-smoke and/or --with-job)" >&2; exit 2 ;;
    esac
done

echo "==> [1/6] Installing recsys + deps"
pip install -q -e .

echo "==> [2/6] Ensuring the teamspace data folder is mounted"
python - <<'EOF'
import os, sys, time, warnings
warnings.filterwarnings("ignore")  # lightning-sdk's "newer version" nag

from recsys.constants import RAW_DATA_DIR, LITDATA_DIR

MOUNT_ROOT = "/teamspace/lightning_storage"

def folder_for(path):
    """Top-level teamspace folder `path` lives in, or None if it's not on the drive."""
    rel = os.path.relpath(os.path.abspath(path), MOUNT_ROOT)
    return None if rel.startswith("..") else rel.split(os.sep)[0]

folders = {f for f in (folder_for(RAW_DATA_DIR), folder_for(LITDATA_DIR)) if f}
if not folders:
    print(f"    Data dirs are overridden outside {MOUNT_ROOT}; nothing to create.")
    sys.exit(0)

missing = [f for f in sorted(folders) if not os.path.isdir(os.path.join(MOUNT_ROOT, f))]
if not missing:
    print(f"    {', '.join(os.path.join(MOUNT_ROOT, f) for f in sorted(folders))} already mounted.")
    sys.exit(0)

from lightning_sdk import Studio
teamspace = Studio().teamspace
for f in missing:
    print(f"    Creating teamspace folder '{f}' in {teamspace.owner.name}/{teamspace.name}")
    teamspace.new_folder(f)

# The mount usually appears within ~60s, no Studio restart needed.
deadline = time.time() + 180
while time.time() < deadline:
    if all(os.path.isdir(os.path.join(MOUNT_ROOT, f)) for f in missing):
        print("    Mounted.")
        sys.exit(0)
    time.sleep(5)
sys.exit(f"    Folder(s) {missing} created but not mounted under {MOUNT_ROOT} after 180s. "
         f"Check `ls {MOUNT_ROOT}`; if your mount root differs, set "
         f"MOVIELENS_DATA_DIR / MOVIELENS_LITDATA_DIR and re-run.")
EOF

echo "==> [3/6] Fetching raw MovieLens 100K"
python training/fetch_data.py

echo "==> [4/6] Building LitData-optimized copy"
python training/optimize_data.py

if [[ $SKIP_SMOKE -eq 1 ]]; then
    echo "==> [5/6] Local smoke test skipped"
else
    echo "==> [5/6] Running local smoke test"
    python training/train_movielens.py --smoke_test
fi

# Opt-in: the local smoke test proves the code runs here, not that it runs as a
# remote Job -- a fresh machine, cwd = studio root, drive over the network. This
# launches one short CPU job and waits for it. Costs a few minutes of compute,
# so it's off by default.
if [[ $WITH_JOB -eq 1 ]]; then
    echo "==> [6/6] Running remote job smoke test"
    python training/smoke_test_job.py
else
    echo "==> [6/6] Remote job smoke test skipped (pass --with-job to run it)"
fi

echo "==> Setup complete."
