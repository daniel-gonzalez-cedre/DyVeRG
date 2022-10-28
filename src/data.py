from os.path import join

import networkx as nx


def load_data(dataname: str = 'email-dnc', lookback: int = 0) -> list[tuple[int, nx.Graph]]:
    assert dataname in ['email-dnc', 'email-eucore', 'facebook-links']

    rootpath = '/Users/danielgonzalez/repos/temporal_VRG/data'
    datapath = f'{rootpath}/{dataname}/{dataname}_processed.edgelist'

    with open(join(rootpath, datapath), 'r') as infile:
        edges = [map(int, line.strip().split(',')) for line in infile]

    edge_dict: dict[int, list[tuple[int, int]]] = {}
    for u, v, t in edges:
        if t in edge_dict:
            edge_dict[t] += [(u, v)]
        else:
            edge_dict[t] = [(u, v)]

    graphs = sorted([(t, nx.Graph(edges, time=t)) for t, edges in edge_dict.items()],
                    key=lambda x: x[0])

    if lookback > 0:
        cum_graphs = [(t, g.copy()) for t, g in graphs]
        for idx, (_, g) in enumerate(graphs):
            cum_edges: set[tuple[int, int]] = {e
                                               for _, h in graphs[max(0, idx - lookback):idx]
                                               for e in h.edges()}
            cum_graphs[idx][1].add_edges_from(cum_edges)

        graphs = cum_graphs

    # if float(nx.__version__[:3]) < 2.4:
    #     giants = [(t, sorted(nx.connected_component_subgraphs(g), key=len, reverse=True)[0])
    #               for t, g in graphs]
    # else:
    #     giants = [(t, g.subgraph(sorted(nx.connected_components(g), key=len, reverse=True)[0]))
    #               for t, g in graphs]

    # return sorted(giants, key=lambda x: x[0])

    return sorted(graphs, key=lambda x: x[0])
