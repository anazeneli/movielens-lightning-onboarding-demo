"""recsys.constants — single source of truth for shared config values.

Every script that reads or writes the MovieLens data (fetch_data.py,
optimize_data.py, movielens_datamodule.py, serving/) imports RAW_DATA_DIR /
LITDATA_DIR from here instead of hardcoding its own default, so changing the
shared-drive root only requires editing one line -- or overriding the two env
vars below, which still take precedence.
"""

import os

RAW_DATA_DIR = os.environ.get("MOVIELENS_DATA_DIR", "/teamspace/lightning_storage/ml-100k")
LITDATA_DIR  = os.environ.get("MOVIELENS_LITDATA_DIR", os.path.join(RAW_DATA_DIR, "ml-100k-optimized"))
