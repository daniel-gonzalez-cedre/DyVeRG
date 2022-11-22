from signal import SIGALRM, ITIMER_REAL, setitimer, signal
from traceback import format_exc

import sys
from contextlib import contextmanager
from os import makedirs, devnull

from typing import Callable, Union
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


def timeout(func: Callable, args: Union[list, tuple] = None, kwargs: dict = None,
            patience: Union[int, float] = 120):
    """
        Runs func on the given arguments until either a result is procured or the patience runs out.

        Positional arguments:
            func: [a] -> [b] = the function to call
            args: Union[list, tuple] = an unpackable of positional arguments to feed to func
            kwargs: dict = a dict of keyword arguments to feed to func

        Keyword arguments:
            patience: Union[int, float] = the amount of seconds to wait for func to produce a result

        Returns:
            a tuple (message, None) if the function times out, otherwise (None, result)
    """
    try:
        signal(SIGALRM, lambda x, y: (_ for _ in ()).throw(TimeoutError))
        setitimer(ITIMER_REAL, patience)
        return None, func(*(args or ()), **(kwargs or {}))
    except TimeoutError:
        return f'{func.__name__} interrupted after {patience} seconds running on {args}', None
    except Exception as e:
        raise Exception(f'{func.__name__} crashed when run on {args}') from e
    finally:
        setitimer(ITIMER_REAL, 0)


def mkdir(path):
    makedirs(path, exist_ok=True)


def find(x: object, iterable: Union[list, dict]):
    references = []

    if isinstance(iterable, list):
        pass

    if isinstance(iterable, list):
        for idx, item in enumerate(iterable):
            if item is x:
                references += [idx]
            elif isinstance(item, (list, dict)):
                if ref := find(x, item):
                    references += [(idx, ref)]  # type: ignore
    elif isinstance(iterable, dict):
        for key, value in iterable.items():
            if value is x:
                references += [key]
            elif isinstance(value, (list, dict)):
                if ref := find(x, value):
                    references += [(key, ref)]  # type: ignore
    return references


def replace(x: object, y: object, iterable: Union[list, dict, set]):
    """
        Replaces every instance of x by y in the iterable collection.
    """
    if isinstance(iterable, list):
        for idx, item in enumerate(iterable):
            if isinstance(item, (list, dict, set)):
                replace(x, y, item)
            elif item is x:
                iterable[idx] = y
    elif isinstance(iterable, dict):
        for key, value in iterable.items():
            if isinstance(value, (list, dict, set)):
                replace(x, y, value)
            elif value is x:
                iterable[key] = y
    elif isinstance(iterable, set):  # python sets cannot contain lists, dicts, or sets
        for element in iterable:
            if element is x:
                iterable.remove(x)
                iterable.add(y)
                break
    else:
        raise NotImplementedError


def node_match_(u, v):
    return (
        (
            u['label'] == v['label']
            if (('label' in u) and ('label' in v))
            else ('label' in u) == ('label' in v)
        ) and (
            u['b_deg'] == v['b_deg']
            if (('b_deg' in u) and ('b_deg' in v))
            else ('b_deg' in u) == ('b_deg' in v)
        )
    )


def edge_match_(e, f):
    return (
        e['weight'] == f['weight']
        if (('weight' in e) and ('weight' in f))
        else ('weight' in e) == ('weight' in f)
    )


def edge_subst_cost_(e, f):
    return (
        abs(e['weight'] - f['weight'])
        if (('weight' in e) and ('weight' in f))
        else 1
    )


def edge_del_cost_(e):
    return (
        e['weight']
        if 'weight' in e
        else 1
    )


def edge_ins_cost_(e):
    return (
        e['weight']
        if 'weight' in e
        else 1
    )


def graph_edit_distance(g1: Graph, g2: Graph,
                        node_match: Callable = node_match_, edge_match: Callable = edge_match_,
                        edge_subst_cost: Callable = edge_subst_cost_,
                        edge_del_cost: Callable = edge_del_cost_, edge_ins_cost: Callable = edge_ins_cost_,
                        timeout: int = 5):
    dist = ged(g1, g2,
               node_match=node_match, edge_match=edge_match,
               edge_subst_cost=edge_subst_cost,
               edge_del_cost=edge_del_cost, edge_ins_cost=edge_ins_cost,
               timeout=timeout)
    return dist if dist is not None else g1.size() + g2.size()
