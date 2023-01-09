from signal import SIGALRM, ITIMER_REAL, setitimer, signal
from typing import Callable, Iterator, Union, Optional

import sys
from contextlib import contextmanager
from os import makedirs, devnull

import math
from collections import defaultdict

from gamma_coding import gamma_coding
from networkx import graph_edit_distance as ged, Graph
import networkx.algorithms.isomorphism as iso


class autodict(defaultdict):
    def __missing__(self, key):
        self.default_factory: Callable

        if self.default_factory is None:
            raise KeyError(key)

        ret = self[key] = self.default_factory(key)
        return ret


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


def once(f):
    def wrapper(*args, **kwargs):  # pylint: disable=inconsistent-return-statements
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper


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


# find all occurrences of an object (e.g., a rule) in a grammar (rule_list, rule_dict, or rule_tree)
def find(x: object, iterable: Union[list, dict]) -> Union[list[int], list[tuple[int, int]]]:
    references = []
    if isinstance(iterable, list):  # either rule_list or rule_tree
        for idx, item in enumerate(iterable):
            if item is x:  # rule_list
                references += [idx]
            elif isinstance(item, (list, dict)):  # rule_tree
                if ref := find(x, item):
                    references += [(idx, ref)]  # type: ignore
    elif isinstance(iterable, dict):  # rule_dict
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


def gamma(x: int) -> int:
    return len(gamma_coding(x))


def graph_mdl(g: Graph) -> int:
    def lu(g: Graph) -> int:
        node_types = set()
        for _, d in g.nodes(data=True):
            if 'label' in d:
                node_types.add('nts')
            else:
                node_types.add('terminal')
        return 1 + len(node_types)

    n = max(1, g.order())
    m = max(1, len(g.edges()))
    types = lu(g)

    # cost to encode each node and differentiate between the types
    dl_nodes = int(math.log2(n) + 1) + int(math.log2(types) + 1) * n

    # cost to encode an edge as an unordered pair of nodes
    dl_edges = (
        int(math.log2(m) + 1) +
        int(math.log2(types) + 1) * sum(2 * gamma(g.number_of_edges(u, v))
                                        for u, v in g.edges())
    )

    return dl_nodes + dl_edges


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
                        patience: int = 120):
    dist = ged(g1, g2,
               node_match=node_match, edge_match=edge_match,
               edge_subst_cost=edge_subst_cost,
               edge_del_cost=edge_del_cost, edge_ins_cost=edge_ins_cost,
               timeout=patience)
    return dist if dist is not None else g1.size() + g2.size()


def graph_isomorphisms(g1: Graph, g2: Graph) -> Iterator[dict]:
    nm = iso.categorical_node_match('label', '')  # does not take into account b_deg on nodes
    em = iso.numerical_edge_match('weight', 1.0)  # pylint: disable=not-callable
    gm = iso.GraphMatcher(g1, g2, node_match=nm, edge_match=em)
    for f in gm.match():
        yield f


def is_graph_isomorphic(g1: Graph, g2: Graph) -> Union[dict, None]:
    for f in graph_isomorphisms(g1, g2):
        return f
    return None


def boundary_edges(g: Graph, nodes: set[int]) -> list[tuple[int, int, Optional[dict]]]:
    nodes = nodes if isinstance(nodes, set) else set(nodes)

    if len(nodes) == g.order():
        return []

    edges = []
    for u in nodes:
        for v in g.neighbors(u):
            if v not in nodes:
                d = g.edges[u, v]
                if 'colors' in d:
                    edge = (u, v, {'colors': d['colors']})
                else:
                    edge = (u, v)
                edges.extend(g.number_of_edges(u, v) * [edge])
    return edges


# def rule_isomorphisms(r1: Rule, r2: Rule) -> Iterator[dict]:
#     if r1.lhs == r2.lhs:
#         for f in graph_isomorphisms(r1.graph, r2.graph):
#             yield f


# def is_rule_isomorphic(r1: Rule, r2: Rule) -> Union[dict, None]:
#     if r1.lhs == r2.lhs:
#         return is_graph_isomorphic(r1.graph, r2.graph)
#     return None
