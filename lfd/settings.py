"""
@created by: heyao
@created at: 2021-12-08 14:30:10
"""
import pathlib
import os

_HOME = pathlib.Path(os.path.expanduser("~"))
PATH_DATASETS = _HOME / "learn_from_datasets"
EMBEDDING_PATH = PATH_DATASETS / "embedding"

OUTPUTS = PATH_DATASETS / "output"
MODELS = PATH_DATASETS / "model"

NAME_DISASTER_TWEETS = "nlp-getting-started"
