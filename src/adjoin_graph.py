import networkx as nx
from tqdm import tqdm

from dyverg.VRG import VRG
from src.utils import silence
from src.decomposition import decompose
from src.adjoin_decomposition import conjoin_grammars
from src.adjoin_rule import domestic, diplomatic, censor_citizen


def update_grammar(grammar: VRG, home_graph: nx.Graph, away_graph: nx.Graph,
                   t1: int, t2: int,
                   mu: int = None, amnesia: bool = False, verbose: bool = False) -> VRG:
    """
        Required arguments:
            grammar = the Vertex Replacement Graph Grammar to update
            home_graph = the old graph that `grammar` was learned on
            away_graph = the new graph providing the updates
            t1 = the timestep associated with `home_graph`
            t2 = the timestep associated with `away_graph`
            signature = one of {i, ja, jb}:
                        i: the independent model
                        ja: the joint model with the `annealing` strategy
                        jb: the joint model with the `branching` strategy

        Optional arguments:
            mu = the `mu` parameter for the VRG
                 by default takes uses `grammar.mu` from the old grammar
            amnesia = whether or not to forget about the node association between the two graphs
                      by default False
            verbose = whether or not to display progress bars and logs
                      by default False
    """
    if mu is None:
        mu = grammar.mu

    if amnesia:
        away_graph = nx.relabel_nodes(away_graph, {v: v + max(home_graph.nodes())
                                                   for v in away_graph.nodes()})

    # signature: set[str] = set(signature.replace('-', ' ').replace('_', ' ').lower())
    # assert signature in ({'i'}, {'i', 'r'},  # independent models
    #                      {'j', 'a'}, {'j', 'r', 'a'},  # joint models: annealing
    #                      {'j', 'b'}, {'j', 'r', 'b'})  # joint models: branching

    assert max(grammar.times) < t2

    charted_grammar: VRG = grammar.copy()
    charted_grammar.ensure(t2)

    nodes_domestic = set(home_graph.nodes())
    for v in away_graph:
        for t in charted_grammar.times:
            if v in charted_grammar.cover[t]:
                nodes_domestic.add(v)
    nodes_foreign = set(away_graph.nodes()) - nodes_domestic

    # node_additions: set[int] = set(away_graph.nodes()) - set(home_graph.nodes())
    node_deletions: set[int] = set(home_graph.nodes()) - set(away_graph.nodes())

    edge_additions: set[tuple[int, int]] = set(away_graph.edges()) - set(home_graph.edges())
    edge_deletions: set[tuple[int, int]] = set(home_graph.edges()) - set(away_graph.edges())

    edges_domestic = {(u, v) for (u, v) in edge_additions
                      if u in nodes_domestic and v in nodes_domestic}
    edges_diplomatic = {(u if u in nodes_domestic else v, v if v not in nodes_domestic else u)  # ensures u ∈ domestic and v ∈ foreign (in that order)
                        for (u, v) in edge_additions
                        if (u in nodes_domestic) != (v in nodes_domestic)}
    edges_foreign = {(u, v) for (u, v) in edge_additions
                     if u not in nodes_domestic and v not in nodes_domestic}

    # edges_domestic: set[tuple[int, int]] = {(u, v)
    #                                         for u, v in edge_additions
    #                                         if (u in home_graph) and (v in home_graph)}
    # edges_diplomatic: set[tuple[int, int]] = {(u if u in home_graph else v, v if v not in home_graph else u)
    #                                           for u, v in edge_additions
    #                                           if bool(u in home_graph) != bool(v in home_graph)}
    # edges_foreign: set[tuple[int, int]] = {(u, v)
    #                                        for u, v in edge_additions
    #                                        if (u not in home_graph) and (v not in home_graph)}

    joint(charted_grammar, nodes_domestic, nodes_foreign,
          edges_domestic, edges_diplomatic, edges_foreign,
          t1, t2, mu, verbose=verbose)

    independent(charted_grammar, nodes_domestic, nodes_foreign,
                edges_domestic, edges_diplomatic, edges_foreign, edge_deletions, node_deletions,
                t1, t2, verbose=verbose)

    # TODO: implement refactoring of rules?
    # if 'r' in signature:
    #     refactor()  # raise NotImplementedError

    for idx, (metarule, _, _) in enumerate(charted_grammar.decomposition):
        metarule.idn = idx
        for rule in metarule:
            rule.idn = idx

    # charted_grammar.compute_rules(t2)
    charted_grammar.compute_levels()
    return charted_grammar


def joint(charted_grammar: VRG, nodes_domestic, nodes_foreign,
          edges_domestic, edges_diplomatic, edges_foreign,
          t1, t2, mu, verbose: bool = False) -> tuple[set, set, set]:
    uncharted_region = nx.Graph()
    uncharted_region.add_edges_from(edges_diplomatic | edges_foreign)
    uncharted_region.remove_nodes_from(nodes_domestic)

    conquered_territories: dict[frozenset[int], nx.Graph] = {}
    uncharted_territories: dict[frozenset[int], nx.Graph] = {frozenset(nodes): uncharted_region.subgraph(nodes)
                                                             for nodes in nx.connected_components(uncharted_region)}

    # incorporate edges from connected components (even if they're not "large enough")
    for nodes, territory in tqdm(uncharted_territories.items(), desc='joint changes', leave=True, disable=(not verbose)):
        with silence():  # suppress progress bars from CNRG
            territory_grammar = decompose(territory, time=t2, mu=mu)

        frontier = {(u, v) for (u, v) in edges_diplomatic if v in territory}

        conjoin_grammars(charted_grammar, territory_grammar, frontier, t1, t2)

        edges_diplomatic -= {(u, v) for u, v in edges_diplomatic if v in territory}
        edges_foreign -= {(u, v) for u, v in edges_foreign if u in territory and v in territory}
        conquered_territories[nodes] = territory

    # TODO: i don't think this block is necessary; think about this
    # remove the incorporated edges from future consideration
    # for territory in conquered_territories.values():
    #     for u, v in territory.edges():
    #         if u in nodes_domestic and v in nodes_domestic:
    #             edges_domestic -= {(u, v)}
    #         elif bool(u in home_graph.nodes()) != bool(v in home_graph.nodes()):
    #             edges_diplomatic -= {(u, v)}
    #         elif u not in home_graph.nodes() and v not in home_graph.nodes():
    #             edges_foreign -= {(u, v)}
    #         else:
    #             raise AssertionError(f'{u}, {v}')


def independent(charted_grammar: VRG, nodes_domestic, nodes_foreign,
                edges_domestic, edges_diplomatic, edges_foreign, edge_deletions, node_deletions,
                t1, t2, verbose: bool = False):
    frontier = edges_domestic | edges_diplomatic

    # perform a BFS on the edge additions
    while len(frontier) > 0:
        # process all of the additions at this stage
        for u, v in tqdm(frontier, desc='adding...', leave=True, disable=(not verbose)):
            if u in nodes_domestic and v in nodes_domestic:
                domestic(charted_grammar, u, v, t1, t2, 'add')
            elif u in nodes_domestic and v not in nodes_domestic:
                diplomatic(charted_grammar, u, v, t1, t2)
            elif u not in nodes_domestic and v in nodes_domestic:  # TODO: this line might be impossible
                diplomatic(charted_grammar, v, u, t1, t2)
            else:
                raise AssertionError(f'{u}, {v}')

        # remove the incorporated edge from future consideration
        for u, v in frontier:
            nodes_domestic |= {u, v}

        frontier = {(u, v) for u, v in edges_foreign
                    if u in nodes_domestic or v in nodes_domestic}
        edges_foreign -= frontier

    # handle the edge deletions
    for u, v in edge_deletions:
        assert u in nodes_domestic and v in nodes_domestic
        domestic(charted_grammar, u, v, t1, t2, 'del')

    # handle the node deletions
    for u in node_deletions:
        censor_citizen(charted_grammar, u, time=t2)


def refactor():
    raise NotImplementedError
