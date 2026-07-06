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

movielens_datamodule.py
    ``MovieLens100K`` — a LightningDataModule that streams MovieLens 100K
    ratings via LitData (``StreamingDataset`` / ``StreamingDataLoader``) from
    a LitData-optimized copy on the shared teamspace drive. Path defaults to
    ``/teamspace/lightning_storage/data/ml-100k-litdata`` and can be overridden
    with the ``MOVIELENS_LITDATA_DIR`` env var. Build that optimized copy once
    with ``python training/optimize_data.py``; ``prepare_data()`` fails fast if
    it's missing rather than rebuilding silently.

Usage
-----
    from recsys.model import TwoTowerModel
    from recsys.movielens_datamodule import MovieLens100K
"""
