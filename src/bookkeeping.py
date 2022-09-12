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
