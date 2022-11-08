from contextlib import contextmanager
import sys
from os import makedirs, devnull
import warnings


@contextmanager
def silence(enabled=True):
    if enabled:
        with warnings.catch_warnings(), open(devnull, "w") as null:
            warnings.filterwarnings(action='ignore')
            old_stdout = sys.stdout
            sys.stdout = null
            old_stderr = sys.stderr
            sys.stderr = null
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    else:
        yield


def mkdir(path):
    makedirs(path, exist_ok=True)
