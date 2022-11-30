# clusters = {}  # type: ignore #  stores the cluster members
# original_graph = None   # to keep track of the original edges covered


def find_boundary_edges(g, vertices):
    """
        Collect all of the boundary edges (i.e., the edges
        that connect the subgraph to the original graph)

        :param g: whole graph
        :param vertices: set of nodes in the subgraph
        :return: boundary edges
    """
    vertices = vertices if isinstance(vertices, set) else set(vertices)

    if len(vertices) == g.order():  # it's the entire node set
        return []

    boundary_edges = []
    for u in vertices:
        for v in g.neighbors(u):
            if v not in vertices:
                d = g.edges[u, v]
                if 'edge_colors' in d.keys():
                    edges = [(u, v, {'edge_colors': d['edge_colors']})] * g.number_of_edges(u, v)
                else:
                    edges = [(u, v)] * g.number_of_edges(u, v)
                boundary_edges.extend(edges)

    return boundary_edges
