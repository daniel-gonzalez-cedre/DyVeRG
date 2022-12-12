import networkx as nx
from tqdm import tqdm

from cnrg.VRG import VRG
from src.utils import silence
from src.decomposition import decompose
from src.adjoin_decomposition import conjoin_grammars
from src.adjoin_rule import mutate_rule_domestic, mutate_rule_diplomatic


def update_grammar(grammar: VRG, home_graph: nx.Graph, away_graph: nx.Graph, time: int, signature: str,
                   mu: int = None, amnesia: bool = False, verbose: bool = False):
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

    edge_additions: set[tuple[int, int]] = set(away_graph.edges()) - set(home_graph.edges())
    # edge_deletions = set(home_graph.edges()) - set(away_graph.edges())

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
        edges_domestic, edges_diplomatic, edges_foreign = joint(charted_grammar, home_graph, edges_domestic, edges_diplomatic, edges_foreign,
                                                                mu, time, approach='branch' if 'b' in signature else 'anneal', verbose=verbose)

    independent(charted_grammar, home_graph, edges_domestic, edges_diplomatic, edges_foreign, time, verbose=verbose)

    # TODO: implement refactoring of rules?
    if 'r' in signature:
        raise NotImplementedError

    charted_grammar.compute_rules()
    charted_grammar.compute_levels()
    return charted_grammar


def joint(charted_grammar: VRG, home_graph: nx.Graph, edges_domestic, edges_diplomatic, edges_foreign,
          mu, time, approach: str = 'branch', verbose: bool = False) -> tuple[set, set, set]:
    uncharted_region = nx.Graph()
    uncharted_region.add_edges_from(edges_diplomatic | edges_foreign)
    uncharted_region.remove_nodes_from(home_graph.nodes())

    conquered_territories: dict[frozenset[int], nx.Graph] = {}
    uncharted_territories: dict[frozenset[int], nx.Graph] = {frozenset(nodes): uncharted_region.subgraph(nodes)
                                                             for nodes in nx.connected_components(uncharted_region)}

    for nodes, territory in tqdm(uncharted_territories.items(), desc='joint changes', leave=True, disable=(not verbose)):
        if territory.order() >= mu:
            with silence():
                territory_grammar = decompose(territory, time=time, mu=mu)

            conquered_territories[nodes] = territory

            frontier = {(u if u in home_graph else v, v if v not in home_graph else u)
                        for (u, v) in edges_diplomatic
                        if v in territory}

            for u, v in frontier:
                assert u in charted_grammar.cover and v in territory_grammar.cover

            conjoin_grammars(charted_grammar, territory_grammar, frontier, time, approach=approach)

            edges_diplomatic -= {(u, v) for u, v in edges_diplomatic if v in territory}
            edges_foreign -= {(u, v) for u, v in edges_foreign if u in territory and v in territory}

    for territory in conquered_territories.values():
        for u, v in territory.edges():
            if u in home_graph.nodes() and v in home_graph.nodes():
                edges_domestic |= {(u, v)}
            elif bool(u in home_graph.nodes()) != bool(v in home_graph.nodes()):
                edges_diplomatic |= {(u, v)}
            elif u not in home_graph.nodes() and v not in home_graph.nodes():
                edges_foreign |= {(u, v)}
            else:
                raise AssertionError(f'{u}, {v}')

    return edges_domestic, edges_diplomatic, edges_foreign


def independent(charted_grammar: VRG, home_graph: nx.Graph, edges_domestic, edges_diplomatic, edges_foreign, time, verbose: bool = False) -> tuple[set, set, set]:
    conquered = set(home_graph.nodes())
    changes = edges_domestic | edges_diplomatic

    # handle the edge additions
    while len(changes) > 0:
        for u, v in tqdm(changes, desc='additions', leave=True, disable=(not verbose)):
            if not charted_grammar.is_edge_connected(u, v):
                charted_grammar.penalty += 1
            else:
                if u in charted_grammar.cover and v in charted_grammar.cover:
                    mutate_rule_domestic(charted_grammar, u, v, time, mode='add')
                elif u in charted_grammar.cover and v not in charted_grammar.cover:
                    mutate_rule_diplomatic(charted_grammar, u, v, time)
                elif u not in charted_grammar.cover and v in charted_grammar.cover:
                    mutate_rule_diplomatic(charted_grammar, v, u, time)
                else:
                    raise AssertionError(f'{u}, {v}')

        for u, v in changes:
            conquered |= {u, v}

        changes = {(u, v) for u, v in edges_foreign
                   if u in conquered or v in conquered}
        edges_foreign -= changes

    # handle the edge deletions
    # for u, v in tqdm(edge_deletions, desc='deletions', leave=True):
    #     charted_grammar = update_rule_domestic(charted_grammar, u, v, 'del')

    # for idx, rule in enumerate(charted_grammar.rule_list):
    #     rule.idn = idx

    return NotImplemented


def refactor():
    return NotImplemented
