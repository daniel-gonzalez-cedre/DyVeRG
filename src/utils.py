from contextlib import contextmanager
import sys
from os import makedirs, devnull

from typing import Callable
from networkx import graph_edit_distance as ged, Graph


@contextmanager
def silence(enabled=True):
    if enabled:
        with open(devnull, "w") as null:
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


def node_matcher(d1, d2):
    return d1.keys() == d2.keys and \
        (not('label' in d1) or d1['label'] == d2['label']) and \
        (not('b_deg' in d1) or d1['b_deg'] == d2['b_deg'])

    # if d1.keys() != d2.keys():
    #     return False

    # if 'label' in d1 and d1['label'] != d2['label']:
    #     return False

    # if 'b_deg' in d1 and d1['b_deg'] != d2['b_deg']:
    #     return False

    # return True


def graph_edit_distance(g1: Graph, g2: Graph, node_match: Callable = node_matcher, timeout: int = 5):
    dist = ged(g1, g2, node_match=node_match, timeout=timeout)
    return dist if dist is not None else g1.size() + g2.size()
