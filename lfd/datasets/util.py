"""
@created by: heyao
@created at: 2021-12-08 14:37:24
"""
import pathlib


def init_path(path):
    if isinstance(path, str):
        return pathlib.Path(path)
    return path
