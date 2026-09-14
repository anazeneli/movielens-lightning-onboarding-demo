# training/fetch_data.py
#
# One-time step: download the raw MovieLens 100K dataset into the shared
# teamspace drive, if it isn't already there. Everything else in this repo
# (optimize_data.py, serving/) expects the raw files directly under
# MOVIELENS_DATA_DIR -- no nested ml-100k/ subfolder.
#
# Run once (skips if already present):
#     python training/fetch_data.py

import hashlib
import os
import shutil
import ssl
import tempfile
import urllib.error
import urllib.request
import zipfile

from recsys.constants import RAW_DATA_DIR as RAW_DIR

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

# Published MD5 of ml-100k.zip. This is what makes the fallback below safe: if
# the TLS chain can't be trusted, the bytes still have to hash to a known value.
DATA_MD5 = "0e33842e24a9c977be4e0107933c0723"


def _download(url, dest):
    """Download `url` to `dest`, verifying the payload hash either way.

    grouplens.org has been serving an expired certificate, which fails the
    normal verified fetch outright. Rather than disable verification globally,
    retry *only* on a certificate error and require the MD5 to match -- trading
    transport trust for payload trust instead of dropping both. Any other
    URLError is a real network failure and propagates.
    """
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
    except urllib.error.URLError as e:
        if not isinstance(e.reason, ssl.SSLCertVerificationError):
            raise
        print(f"WARNING: certificate verification failed for {url} ({e.reason.verify_message}).\n"
              f"         Retrying unverified; the download must match MD5 {DATA_MD5}.")
        with urllib.request.urlopen(url, timeout=120, context=ssl._create_unverified_context()) as r:
            data = r.read()

    digest = hashlib.md5(data).hexdigest()
    if digest != DATA_MD5:
        raise RuntimeError(
            f"Checksum mismatch for {url}: got {digest}, expected {DATA_MD5}. "
            f"Refusing to use this download."
        )
    with open(dest, "wb") as f:
        f.write(data)
    print(f"Downloaded {len(data)} bytes, MD5 verified.")


def main():
    marker = os.path.join(RAW_DIR, "u.data")
    if os.path.exists(marker):
        print(f"Raw data already present at {RAW_DIR} -- nothing to do.")
        return

    os.makedirs(RAW_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "ml-100k.zip")
        print(f"Downloading {DATA_URL} -> {zip_path}")
        _download(DATA_URL, zip_path)

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
