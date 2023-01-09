import networkx as nx
from tqdm import tqdm

from cnrg.VRG import VRG
from src.utils import silence
from src.decomposition import decompose
from src.adjoin_decomposition import conjoin_grammars
from src.adjoin_rule import domestic, diplomatic, remove_citizen


def update_grammar(grammar: VRG, home_graph: nx.Graph, away_graph: nx.Graph,
                   t1: int, t2: int, signature: str, mu: int = None, amnesia: bool = False,
                   verbose: bool = False):
    """
        Required arguments:
            signature =

        Optional arguments:
    """
    if mu is None:
        mu = grammar.mu

    if amnesia:
        away_graph = nx.relabel_nodes(away_graph, {v: v + max(home_graph.nodes())
                                                   for v in away_graph.nodes()})

    signature: set[str] = set(signature.replace('-', ' ').replace('_', ' ').lower())
    assert signature in ({'i'}, {'i', 'r'},  # independent models
                         {'j', 'a'}, {'j', 'r', 'a'},  # joint models: annealing
                         {'j', 'b'}, {'j', 'r', 'b'})  # joint models: branching

    charted_grammar: VRG = grammar.copy()
    charted_grammar.ensure(t1, t2)

    # node_additions: set[int] = set(away_graph.nodes()) - set(home_graph.nodes())
    node_deletions: set[int] = set(home_graph.nodes()) - set(away_graph.nodes())

    edge_additions: set[tuple[int, int]] = set(away_graph.edges()) - set(home_graph.edges())
    edge_deletions: set[tuple[int, int]] = set(home_graph.edges()) - set(away_graph.edges())

    edges_domestic: set[tuple[int, int]] = {(u, v)
                                            for u, v in edge_additions
                                            if (u in home_graph) and (v in home_graph)}
    edges_diplomatic: set[tuple[int, int]] = {(u if u in home_graph else v, v if v not in home_graph else u)
                                              for u, v in edge_additions
                                              if bool(u in home_graph) != bool(v in home_graph)}
    edges_foreign: set[tuple[int, int]] = {(u, v)
                                           for u, v in edge_additions
                                           if (u not in home_graph) and (v not in home_graph)}

    if 'j' in signature:
        edges_domestic, edges_diplomatic, edges_foreign = joint(charted_grammar, home_graph,
                                                                edges_domestic, edges_diplomatic, edges_foreign,
                                                                t1, t2, mu, approach=('branch' if 'b' in signature else 'anneal'), verbose=verbose)

    independent(charted_grammar, home_graph,
                edges_domestic, edges_diplomatic, edges_foreign, edge_deletions, node_deletions,
                t1, t2, verbose=verbose)

    # TODO: implement refactoring of rules?
    if 'r' in signature:
        refactor()  # raise NotImplementedError

    for idx, (metarule, _, _) in enumerate(charted_grammar.decomposition):
        metarule.idn = idx

    charted_grammar.compute_rules(t2)
    charted_grammar.compute_levels()
    charted_grammar.times += [t2]
    return charted_grammar


def joint(charted_grammar: VRG, home_graph: nx.Graph,
          edges_domestic, edges_diplomatic, edges_foreign,
          t1, t2, mu, approach: str = 'branch', verbose: bool = False) -> tuple[set, set, set]:
    uncharted_region = nx.Graph()
    uncharted_region.add_edges_from(edges_diplomatic | edges_foreign)
    uncharted_region.remove_nodes_from(home_graph.nodes())

    conquered_territories: dict[frozenset[int], nx.Graph] = {}
    uncharted_territories: dict[frozenset[int], nx.Graph] = {frozenset(nodes): uncharted_region.subgraph(nodes)
                                                             for nodes in nx.connected_components(uncharted_region)}

    # incorporate edges from "large-enough" connected components
    for nodes, territory in tqdm(uncharted_territories.items(), desc='joint changes', leave=True, disable=(not verbose)):
        if territory.order() >= mu:
            with silence():  # suppress progress bars from CNRG
                territory_grammar = decompose(territory, time=t2, mu=mu)

            frontier = {(u if u in home_graph else v, v if v not in home_graph else u)
                        for (u, v) in edges_diplomatic
                        if v in territory}

            for u, v in frontier:
                assert u in charted_grammar.cover[t2] and v in territory_grammar.cover[t2]

            conjoin_grammars(charted_grammar, territory_grammar, frontier, t1, t2, approach=approach)

            edges_diplomatic -= {(u, v) for u, v in edges_diplomatic if v in territory}
            edges_foreign -= {(u, v) for u, v in edges_foreign if u in territory and v in territory}
            conquered_territories[nodes] = territory

    # remove the incorporated edges from future consideration
    for territory in conquered_territories.values():
        for u, v in territory.edges():
            if u in home_graph.nodes() and v in home_graph.nodes():
                edges_domestic -= {(u, v)}
            elif bool(u in home_graph.nodes()) != bool(v in home_graph.nodes()):
                edges_diplomatic -= {(u, v)}
            elif u not in home_graph.nodes() and v not in home_graph.nodes():
                edges_foreign -= {(u, v)}
            else:
                raise AssertionError(f'{u}, {v}')

    return edges_domestic, edges_diplomatic, edges_foreign


def independent(charted_grammar: VRG, home_graph: nx.Graph,
                edges_domestic, edges_diplomatic, edges_foreign, edge_deletions, node_deletions,
                t1, t2, verbose: bool = False):
    conquered = set(home_graph.nodes())
    add_frontier = edges_domestic | edges_diplomatic

    # perform a BFS on the edge additions
    while len(add_frontier) > 0:
        # process all of the additions at this stage
        for u, v in tqdm(add_frontier, desc='adding...', leave=True, disable=(not verbose)):
            if u in charted_grammar.cover[t2] and v in charted_grammar.cover[t2]:
                domestic(charted_grammar, u, v, t1, t2, 'add')
            elif u in charted_grammar.cover[t2] and v not in charted_grammar.cover[t2]:
                diplomatic(charted_grammar, u, v, t1, t2)
            elif u not in charted_grammar.cover[t2] and v in charted_grammar.cover[t2]:
                diplomatic(charted_grammar, v, u, t1, t2)
            else:
                raise AssertionError(f'{u}, {v}')

        # remove the incorporated edge from future consideration
        for u, v in add_frontier:
            conquered |= {u, v}

        add_frontier = {(u, v) for u, v in edges_foreign
                        if u in conquered or v in conquered}
        edges_foreign -= add_frontier

    # handle the edge deletions
    for u, v in edge_deletions:
        domestic(charted_grammar, u, v, t1, t2, 'del')

    # handle the node deletions
    for u in node_deletions:
        remove_citizen(charted_grammar, u, time=t2)


def refactor():
    raise NotImplementedError
