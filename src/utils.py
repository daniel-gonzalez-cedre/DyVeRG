from contextlib import contextmanager
import sys
import os
import warnings


@contextmanager
def silence(enabled=True):
    if enabled:
        with warnings.catch_warnings(), open(os.devnull, "w") as devnull:
            warnings.filterwarnings(action='ignore')
            old_stdout = sys.stdout
            sys.stdout = devnull
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    else:
        yield
