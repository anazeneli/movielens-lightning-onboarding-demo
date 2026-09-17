"""recsys.constants — single source of truth for shared config values.

Every script that reads or writes the MovieLens data (fetch_data.py,
optimize_data.py, movielens_datamodule.py, serving/) imports RAW_DATA_DIR /
LITDATA_DIR from here instead of hardcoding its own default, so changing the
shared-drive root only requires editing one line -- or overriding the two env
vars below, which still take precedence.
"""

import os

RAW_DATA_DIR = os.environ.get("MOVIELENS_DATA_DIR", "/teamspace/lightning_storage/data/ml-100k")
LITDATA_DIR  = os.environ.get("MOVIELENS_LITDATA_DIR", "/teamspace/lightning_storage/data/ml-100k-litdata")


def registry_name(experiment_name: str) -> str:
    """Flatten a (possibly nested) experiment name into a model-registry name.

    Experiment names may contain "/" -- that is exactly what renders folder
    hierarchy in the experiment manager. The *model registry* parses "/" as its
    own `owner/teamspace/model_name` delimiter, so a nested experiment name
    reaches it as 6 parts instead of 3 and the upload fails with:

        ValueError: Model name must be in the format `organization/teamspace/model_name`

    So the two names are deliberately decoupled: the experiment keeps its
    slashes, its checkpoint registers under this flattened form. Training and
    serving both go through this function, so they always agree on the mapping.

    A name that is already flat passes through unchanged, so checkpoints
    registered before the split keep resolving under the same string.
    """
    return experiment_name.strip("/").replace("/", "-")
