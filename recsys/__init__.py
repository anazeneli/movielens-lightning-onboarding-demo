"""recsys — shared library for the MovieLens two-tower recommender.

This package holds the code that BOTH the training and serving layers import,
so the model definition and data pipeline have a single source of truth.

Modules
-------
model.py
    ``TwoTowerModel`` — a LightningModule with separate user and item
    embedding tables whose dot product scores a (user, item) pair. Trained on
    implicit feedback (rating >= 4 → positive); logs accuracy / precision /
    recall / average-precision and a precision-recall-curve artifact.

constants.py
    ``RAW_DATA_DIR`` / ``LITDATA_DIR`` — single source of truth for the
    shared-drive paths. Every script that touches the MovieLens data
    (``training/fetch_data.py``, ``training/optimize_data.py``,
    ``movielens_datamodule.py``, ``serving/``) imports these instead of
    hardcoding its own default, so changing the shared-drive root only means
    editing this one file (or overriding ``MOVIELENS_DATA_DIR`` /
    ``MOVIELENS_LITDATA_DIR``, which still take precedence).

movielens_datamodule.py
    ``MovieLens100K`` — a LightningDataModule that streams MovieLens 100K
    ratings via LitData (``StreamingDataset`` / ``StreamingDataLoader``) from
    the LitData-optimized copy at ``constants.LITDATA_DIR``. Build it once
    with ``python training/optimize_data.py``; ``prepare_data()`` fails fast if
    it's missing rather than rebuilding silently.

Usage
-----
    from recsys.model import TwoTowerModel
    from recsys.movielens_datamodule import MovieLens100K
"""
