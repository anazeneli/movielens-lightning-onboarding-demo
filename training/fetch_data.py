# training/fetch_data.py
#
# One-time step: download the raw MovieLens 100K dataset into the shared
# teamspace drive, if it isn't already there. Everything else in this repo
# (optimize_data.py, serving/) expects the raw files directly under
# MOVIELENS_DATA_DIR -- no nested ml-100k/ subfolder.
#
# Run once (skips if already present):
#     python training/fetch_data.py

import os
import shutil
import tempfile
import urllib.request
import zipfile

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
RAW_DIR = os.environ.get("MOVIELENS_DATA_DIR", "/teamspace/lightning_storage/data/ml-100k")


def main():
    marker = os.path.join(RAW_DIR, "u.data")
    if os.path.exists(marker):
        print(f"Raw data already present at {RAW_DIR} -- nothing to do.")
        return

    os.makedirs(RAW_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "ml-100k.zip")
        print(f"Downloading {DATA_URL} -> {zip_path}")
        urllib.request.urlretrieve(DATA_URL, zip_path)

        print(f"Extracting into {RAW_DIR}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        # The archive unzips into an "ml-100k/" folder; move its contents up
        # into RAW_DIR directly so paths match what the rest of the repo expects.
        extracted = os.path.join(tmp, "ml-100k")
        for name in os.listdir(extracted):
            shutil.move(os.path.join(extracted, name), os.path.join(RAW_DIR, name))

    print(f"Done. Raw MovieLens 100K files are in {RAW_DIR}")


if __name__ == "__main__":
    main()
