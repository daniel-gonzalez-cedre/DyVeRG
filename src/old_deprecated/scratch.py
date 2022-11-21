def find_cover(u, grammar):
    which_u = grammar.which_rule_source[u]  # the reference to the node u in the rule that covers u
    idx_u = grammar.rule_source[u]  # the index in the tree of the rule that covers u
    rule = grammar.rule_tree[idx_u][0]  # the actual rule that covers u
    return rule, which_u
    
def find_common_ancestor(u, v, grammar):
    idx_u = grammar.rule_source[u]  # index of u in the rule tree
    idx_v = grammar.rule_source[v]  # index of v in the rule tree
    u_anc = []  # ancestor rules of u
    v_anc = []  # ancestor rules of v

    # walk all the way up the ancestor chain for u until we reach the root
    while idx_u is not None:
        u_anc.append(idx_u)
        idx_u = grammar.rule_tree[idx_u][1]

    # walk up the ancestor chain for v until we intersect the chain for u
    while idx_v not in u_anc:
        v_anc.append(idx_v)
        idx_v = grammar.rule_tree[idx_v][1]

        # if we somehow did not intersect, panic
        if idx_v is None:
            return np.log(0)
        
    # now idx_v should refer to the common ancestor of both u and v

    parent_rule = grammar.rule_tree[idx_v][0]
    child_rule_u = None
    child_rule_v = None
    which_child_u = None
    which_child_v = None

    if len(v_anc) > 0:
        child_rule_v = grammar.rule_tree[v_anc[-1]][0]

        if v_anc[-1] in u_anc and u_anc.index(v_anc[-1]) > 0:
            child_rule_u = grammar.rule_tree[u_anc.index(v_anc[-1]) - 1][0]

    if child_rule_v is not None:
        which_child_v = grammar.rule_tree[v_anc[-1]][2]
    else:
        which_child_v = grammar.which_rule_source[v]

    if child_rule_u is not None:
        which_child_u = grammar.rule_tree[u_anc.index(v_anc[-1])-1][2]
    else:
        which_child_u = grammar.which_rule_source[u]

    parent_rule_c = copy.deepcopy(parent_rule)
    parent_rule_c.graph = copy.deepcopy(parent_rule.graph)

    if which_child_u >= len(parent_rule_c.graph.nodes()):
        # print('u', u)
        # print(which_child_u, len(parent_rule_c.graph.nodes()))
        # print(parent_rule_c)
        return u, which_child_u, len(parent_rule_c.graph.nodes())
        # return np.log(0)
    
    if which_child_v >= len(parent_rule_c.graph.nodes()):
        # print('v', v)
        # print(which_child_v, len(parent_rule_c.graph.nodes()))
        # print(parent_rule_c)
        return v, which_child_v, len(parent_rule_c.graph.nodes())
        # return np.log(0)
    
    parent_rule_c.graph.add_edge( list(parent_rule_c.graph.nodes())[which_child_u], list(parent_rule_c.graph.nodes())[which_child_v])

    return parent_rule_c, which_child_u, which_child_v

# takes a graph and extracts a grammar
# also returns the associated dendrogram and rule extraction sequence
def get_decomposition(g, clustering='leiden', gtype='mu_level_dl', name='', mu=4):
    if not isinstance(g, LightMultiGraph):
        g = convert_LMG(g)
        
    if clustering == 'leiden':
        clusters = leiden(g)
    elif clustering == 'louvain':
        # check that this works
        raise NotImplementedError
        clusters = louvain(g)
    else:
        raise NotImplementedError
        
    dendrogram = create_tree(clusters)

    vrg = VRG(clustering=clustering, \
              type=gtype, \
              name=name, \
              mu=mu)

    extractor = MuExtractor(g=g.copy(), \
                            type=gtype, \
                            grammar=vrg, \
                            mu=mu, \
                            root=dendrogram.copy())

    extractor.generate_grammar()
    ex_sequence = extractor.extracted_sequence
    grammar = extractor.grammar
    
    return ex_sequence, extractor, grammar
