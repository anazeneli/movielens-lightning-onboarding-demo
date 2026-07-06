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
    ``MovieLens100K`` — a LightningDataModule that reads the MovieLens 100K
    ratings from the shared teamspace drive and yields train/val dataloaders.
    Data path defaults to ``/teamspace/lightning_storage/data/ml-100k`` and can
    be overridden with the ``MOVIELENS_DATA_DIR`` env var. ``prepare_data()``
    fails fast if the data is missing rather than downloading silently.

Usage
-----
    from recsys.model import TwoTowerModel
    from recsys.movielens_datamodule import MovieLens100K
"""
