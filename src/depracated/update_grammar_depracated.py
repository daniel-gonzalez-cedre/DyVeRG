import networkx as nx
from tqdm import tqdm

from cnrg.VRG import VRG
from utils import silence
from bookkeeping import decompose_component
from grammar_transitions import graft_grammars


# deprecated
def update_chunk(grammar: VRG, curr_graph: nx.Graph, next_graph: nx.Graph):
    mu = 4
    updated_grammar = grammar.copy()
    grammar.init_temporal_matrix()
    # seen = set(curr_graph.nodes())

    edge_additions = set(next_graph.edges()) - set(curr_graph.edges())
    # edge_deletions = set(curr_graph.edges()) - set(next_graph.edges())

    # case1 = {(u, v) for u, v in edge_additions
    #          if u in curr_graph.nodes() and v in curr_graph.nodes()}
    case2 = {(u, v) for u, v in edge_additions
             if bool(u in curr_graph.nodes()) != bool(v in curr_graph.nodes())}
    case3 = {(u, v) for u, v in edge_additions
             if u not in curr_graph.nodes() and v not in curr_graph.nodes()}

    unexplored_region = nx.Graph()
    unexplored_region.add_edges_from(case2 | case3)

    new_nodes = list(nx.connected_components(unexplored_region))
    unexplored_node_components = [(nodes, unexplored_region.subgraph(nodes)) for nodes in new_nodes]

    for nodes, component in unexplored_node_components:
        if component.order() > mu:  # process the update as a chunk of size at least μ
            with silence():
                component_grammar = decompose_component(component, mu=mu)
                # frontier = {(u if u not in component else v, v if v in component else u)
                #             for (u, v) in case2 if (u in component) or (v in component)}

                # first coordinate is in the old part of the graph
                # second coordinate is in the new part of the graph
                frontier = {(u if u in curr_graph else v, v if v not in curr_graph else u)
                            for (u, v) in case2 if (u in component) or (v in component)}

                updated_grammar = graft_grammars(updated_grammar, component_grammar, frontier)
            return updated_grammar

    raise NotImplementedError


# deprecated
def update_independent(grammar: VRG, curr_graph: nx.Graph, next_graph: nx.Graph):
    grammar = grammar.copy()

    if not grammar.temporal_matrix():
        grammar.init_temporal_matrix()

    seen = set(curr_graph.nodes())

    for v in seen:
        if v not in grammar.which_rule_source:
            print(v)
            raise AssertionError

    edge_additions = set(next_graph.edges()) - set(curr_graph.edges())
    edge_deletions = set(curr_graph.edges()) - set(next_graph.edges())

    case1 = {(u, v) for u, v in edge_additions
             if u in curr_graph.nodes() and v in curr_graph.nodes()}
    case2 = {(u, v) for u, v in edge_additions
             if bool(u in curr_graph.nodes()) != bool(v in curr_graph.nodes())}
    case3 = {(u, v) for u, v in edge_additions
             if u not in curr_graph.nodes() and v not in curr_graph.nodes()}

    changes = case1 | case2

    count = 0
    while len(changes) > 0:
        for u, v in tqdm(changes, desc=f'additions: {count}', leave=True):
            if u in seen and v in seen:
                # print('case1')
                grammar = update_rule_case1(u, v, grammar, 'add')
            elif u in seen and v not in seen:
                # print('case2.1')
                grammar = update_rule_case2(u, v, grammar, 'add')
            elif u not in seen and v in seen:
                # print('case2.2')
                grammar = update_rule_case2(v, u, grammar, 'add')
            elif u not in seen and v not in seen:
                raise AssertionError
            else:
                raise AssertionError

        for u, v in changes:
            seen |= {u, v}

        changes = {(u, v) for u, v in case3 if u in seen or v in seen}
        case3 -= changes
        count += 1

    for u, v in tqdm(edge_deletions, desc='deletions', leave=True):
        grammar = update_rule_case1(u, v, grammar, 'del')

    return grammar
