import sys
import copy
import numpy as np
import networkx as nx
# import networkx.algorithms.isomorphism as iso
from tqdm import tqdm

sys.path.append('../../cnrg')
sys.path.append('../../cnrg/utils')

from cnrg.VRG import VRG
from cnrg.Tree import create_tree  # type: ignore
from cnrg.LightMultiGraph import LightMultiGraph as LMG
from cnrg.extract import MuExtractor
from cnrg.partitions import leiden, louvain, get_random_partition


def convert_LMG(g: nx.Graph):
    g_lmg = LMG()
    g_lmg.add_nodes_from(g.nodes())
    g_lmg.add_edges_from(g.edges())
    return g_lmg


def decompose(g: nx.Graph, clustering: str = 'leiden', gtype: str = 'mu_level_dl', name: str = '', mu: int = 4):
    if not isinstance(g, LMG):
        g = convert_LMG(g)

    if clustering == 'leiden':
        clusters = leiden(g)
    elif clustering == 'louvain':
        clusters = louvain(g)
    else:
        raise NotImplementedError

    dendrogram = create_tree(clusters)

    vrg = VRG(clustering=clustering,
              type=gtype,
              name=name,
              mu=mu)

    extractor = MuExtractor(g=g.copy(),
                            type=gtype,
                            grammar=vrg,
                            mu=mu,
                            root=dendrogram.copy())

    extractor.generate_grammar()
    # ex_sequence = extractor.extracted_sequence
    grammar = extractor.grammar

    return grammar
    # return extractor, grammar


def ancestor(u: int, grammar: VRG):
    which_u = grammar.which_rule_source[u]  # points to which node in the rule's RHS corresponds to u
    which_parent = grammar.rule_source[u]  # points to which entry in rule_tree contains this rule
    parent_rule = grammar.rule_tree[which_parent][0]  # the rule in question

    return parent_rule, which_parent, which_u


def common_ancestor(u: int, v: int, grammar: VRG):
    ind_u = grammar.rule_source[u]
    ind_v = grammar.rule_source[v]

    u_anc = []
    v_anc = []

    # trace the ancestral lineage of u all the way to the root
    while ind_u is not None:
        u_anc.append(ind_u)
        ind_u = grammar.rule_tree[ind_u][1]

    # trace the ancestral lineage of v until it intersects u's lineage
    while ind_v not in u_anc:
        v_anc.append(ind_v)
        ind_v = grammar.rule_tree[ind_v][1]

        if ind_v is None:
            return np.log(0)  # somehow the two paths did not cross

    parent_idx = ind_v

    parent_rule = grammar.rule_tree[parent_idx][0]
    # child_rule_u = None
    # child_rule_v = None
    which_child_u = None
    which_child_v = None

    if len(v_anc) > 0:
        # child_rule_v = grammar.rule_tree[v_anc[-1]][0]
        which_child_v = grammar.rule_tree[v_anc[-1]][2]
    else:
        # child_rule_v = grammar.rule_tree[grammar.rule_source[v]][0]
        which_child_v = grammar.which_rule_source[v]

    if u_anc.index(parent_idx) == 0:
        # child_rule_u = grammar.rule_tree[grammar.rule_source[u]][0]
        which_child_u = grammar.which_rule_source[u]
    else:
        # child_rule_u = grammar.rule_tree[u_anc[u_anc.index(parent_idx) - 1]][0]
        which_child_u = grammar.rule_tree[u_anc[u_anc.index(parent_idx) - 1]][2]

    parent_rule_c = copy.deepcopy(parent_rule)
    parent_rule_c.graph = copy.deepcopy(parent_rule.graph)

    #
    # parent_rule_c.graph.add_edge(list(parent_rule_c.graph.nodes())[which_child_u],
    #                              list(parent_rule_c.graph.nodes())[which_child_v])

    return parent_rule_c, parent_idx, which_child_u, which_child_v


def update_grammar_independent(grammar: VRG, graph1: nx.Graph, graph2: nx.Graph):
    grammar = grammar.copy()
    grammar.init_transition_matrix()
    seen = set(graph1.nodes())

    for v in seen:
        if v not in grammar.which_rule_source:
            print(v)
            raise AssertionError

    edge_additions = set(graph2.edges()) - set(graph1.edges())
    edge_deletions = set(graph1.edges()) - set(graph2.edges())

    case1 = {(u, v) for u, v in edge_additions
             if u in graph1.nodes() and v in graph1.nodes()}
    case2 = {(u, v) for u, v in edge_additions
             if bool(u in graph1.nodes()) != bool(v in graph1.nodes())}
    case3 = {(u, v) for u, v in edge_additions
             if u not in graph1.nodes() and v not in graph1.nodes()}

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
        # print('!!!!!!!!!', grammar.rule_source[267])

        for u, v in changes:
            seen |= {u, v}

        changes = {(u, v) for u, v in case3 if u in seen or v in seen}
        case3 -= changes
        count += 1

    # need to implement the updated deletions code
    for u, v in tqdm(edge_deletions, desc='deletions', leave=True):
        grammar = update_rule_case1(u, v, grammar, 'del')

    return grammar


# at time t: node u exists, node v also exists
def update_rule_case1(u: int, v: int, grammar: VRG, mode: str):
    parent_rule, which_parent, which_u, which_v = common_ancestor(u, v, grammar)

    try:
        ancestor_u = list(parent_rule.graph.nodes())[which_u]
        ancestor_v = list(parent_rule.graph.nodes())[which_v]
    except IndexError as e:
        print(e)
        print(f'mode:\t{mode}')
        print(f'which_u:\t{which_u}')
        print(f'which_v:\t{which_v}')
        print(f'len:\t{len(list(parent_rule.graph.nodes()))}')
        # print(f'size:\t{len(list(parent_rule.graph.edges()))}')
        print(f'parent_rule:\t{parent_rule}')
        print(f'parent_rule.graph:\t{parent_rule.graph}')
        print(f'parent_rule.graph.nodes:\t{parent_rule.graph.nodes(data=True)}')
        print(f'parent_rule.graph.edges:\t{parent_rule.graph.edges(data=True)}')
        print(f'u: {u},\tv: {v}')
        # print(f'canon_graph.nodes:\t{parent_rule.canon_graph.nodes(data=True)}')
        # print(f'canon_graph.edges:\t{parent_rule.canon_graph.edges(data=True)}')
        # print(f'{parent_rule.canon_graph.order(), parent_rule.canon_graph.size()}')
        # print(f'canon_matrix:\t{parent_rule.canon_matrix}')
        # print(f'{parent_rule.canon_matrix.shape}')
        raise Exception('?? wew ??')

    new_rule = copy.deepcopy(parent_rule)

    if mode == 'add':
        new_rule.graph.add_edge(ancestor_u, ancestor_v)
    elif mode == 'del':
        try:
            new_rule.graph.edges[ancestor_u, ancestor_v]['weight'] -= 1
            if new_rule.graph.edges[ancestor_u, ancestor_v]['weight'] == 0:
                new_rule.graph.remove_edge(ancestor_u, ancestor_v)
        except Exception:
            print(f'the connection {ancestor_u} --- {ancestor_v} cannot be further severed')
    else:
        raise AssertionError('<<mode>> must be either "add" or "del" ' +
                             f'found mode={mode} instead')

    rule, parent_idx, new_idx = incorporate_new_rule(grammar, parent_rule, new_rule, which_parent)

    # parent_idx = grammar.rule_list.index(parent_rule)
    # new_idx = grammar.rule_list.index(new_rule)

    # grammar.transition_matrix[parent_idx, parent_idx] = max(0, grammar.transition_matrix[parent_idx, parent_idx] - 1)
    grammar.transition_matrix[parent_idx, new_idx] += 1

    return grammar


# at time t: node u exists, node v does not exist
def update_rule_case2(u: int, v: int, grammar: VRG, mode: str):
    parent_rule, which_parent, which_u = ancestor(u, grammar)
    ancestor_u = list(parent_rule.graph.nodes())[which_u]

    new_rule = copy.deepcopy(parent_rule)
    new_rule.graph.add_node(v, b_deg=0)  # TODO: does this b_deg make sense?

    if mode == 'add':
        new_rule.graph.add_edge(ancestor_u, v)
    elif mode == 'del':
        # is this even possible? i don't think so
        raise Exception('!! PANIC !!')
        # new_rule.graph.add_edge(ancestor_u, v)
    else:
        raise AssertionError('<<mode>> must be either "add" or "del" ' +
                             f'found mode={mode} instead')

    grammar.rule_source[v] = grammar.rule_source[u]  # rule_source now points to the same location in the rule_tree for u and v
    # grammar.which_rule_source[u] = list(rule.graph.nodes()).index(u)  # recompute the pointer for u in the rule's RHS (THIS LINE MIGHT NOT MAKE SENSE!!)
    grammar.which_rule_source[v] = list(new_rule.graph.nodes()).index(v)  # compute the pointer for v in the NEW rule's RHS

    rule, parent_idx, new_idx = incorporate_new_rule(grammar, parent_rule, new_rule, which_parent)

    # parent_idx = grammar.rule_list.index(parent_rule)
    # new_idx = grammar.rule_list.index(new_rule)

    # grammar.transition_matrix[parent_idx, parent_idx] = max(0, grammar.transition_matrix[parent_idx, parent_idx] - 1)
    grammar.transition_matrix[parent_idx, new_idx] += 1

    return grammar


def incorporate_new_rule(grammar, parent_rule, new_rule, which_parent, mode='hash'):
    assert mode in ['iso', 'hash', 'canon']

    if mode == 'iso':  # this one is working; the other one isn't (out of bounds index `which_u`, don't know why yet)
        parent_idx = grammar.rule_list.index(parent_rule)

        if new_rule in grammar.rule_dict[new_rule.lhs]:
            new_idx = grammar.rule_list.index(new_rule)
            grammar.rule_list[new_idx].frequency += 1
            grammar.rule_tree[which_parent][0] = grammar.rule_list[new_idx]
            return grammar.rule_list[new_idx], parent_idx, new_idx

    elif mode == 'hash':
        for parent_idx, other_rule in enumerate(grammar.rule_list):
            if parent_rule.hash_equals(other_rule):
                break

        for new_idx, other_rule in enumerate(grammar.rule_list):
            if new_rule.hash_equals(other_rule):
                grammar.rule_list[new_idx].frequency += 1
                grammar.rule_tree[which_parent][0] = grammar.rule_list[new_idx]

                return grammar.rule_list[new_idx], parent_idx, new_idx

    elif mode == 'canon':
        for parent_idx, other_rule in enumerate(grammar.rule_list):
            if parent_rule.canon_equals(other_rule):
                break

        # for index, other_rule in enumerate(grammar.rule_dict[new_rule.lhs]):
        for new_idx, other_rule in enumerate(grammar.rule_list):
            if new_rule.canon_equals(other_rule):
                grammar.rule_list[new_idx].frequency += 1
                grammar.rule_tree[which_parent][0] = grammar.rule_list[new_idx]

                #for idx, child_rule, source_idx, _ in enumerate(grammar.rule_tree):
                #    if grammar.rule_tree[source_idx][0].canon_equals(other_rule):
                #        grammar.rule_tree[idx][2] = 
                #    pass

                return grammar.rule_list[new_idx], parent_idx, new_idx

    else:
        raise NotImplementedError(f'mode {mode} not recognized')

    # if the rule is new
    new_idx = len(grammar.rule_list)  # the index corresponding to the new rule
    grammar.transition_matrix = np.append(grammar.transition_matrix,
                                          np.zeros((1, new_idx)),
                                          axis=0)
    grammar.transition_matrix = np.append(grammar.transition_matrix,
                                          np.zeros((new_idx + 1, 1)),
                                          axis=1)

    grammar.rule_list.append(new_rule)
    grammar.rule_tree[which_parent][0] = new_rule
    #grammar.rule_tree[which_parent][2] =  HERE

    if new_rule.lhs in grammar.rule_dict:
        grammar.rule_dict[new_rule.lhs].append(new_rule)
    else:
        grammar.rule_dict[new_rule.lhs] = [new_rule]

    return grammar.rule_list[new_idx], parent_idx, new_idx



