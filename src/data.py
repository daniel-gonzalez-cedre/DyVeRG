from os import getcwd
from os.path import join

import git
import networkx as nx


# load an arbitrary edgelist, optionally with timestamped edges
def read_edgelist(filepath: str, delimiter: str = ' ', times: bool = True) -> nx.Graph:
    # return nx.convert_node_labels_to_integers(nx.read_edgelist(filepath, delimiter=delimiter))  # node labels must be integers
    g = nx.read_edgelist(filepath,
                         delimiter=delimiter,
                         create_using=nx.MultiGraph,
                         nodetype=int,
                         data=[('time', int)] if times else True)
    return nx.convert_node_labels_to_integers(g)


# load one of the standard datasets
def load_data(dataname: str, mode: str) -> list[tuple[int, nx.Graph]]:
    dataname = dataname.lower().strip()
    mode = mode.lower().strip()
    assert dataname in ('email-dnc', 'email-eucore', 'email-enron', 'facebook-links')
    assert mode in ('full', 'pruned')

    rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    datapath = f'{rootpath}/data/{dataname}/{dataname}_{mode}.edgelist'

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

    # no more lookback nonsense
    # if lookback > 0:
    #     cum_graphs = [(t, g.copy()) for t, g in graphs]
    #     for idx, (_, g) in enumerate(graphs):
    #         cum_edges: set[tuple[int, int]] = {e
    #                                            for _, h in graphs[max(0, idx - lookback):idx]
    #                                            for e in h.edges()}
    #         cum_graphs[idx][1].add_edges_from(cum_edges)

    #     graphs = cum_graphs

    # if float(nx.__version__[:3]) < 2.4:
    #     giants = [(t, sorted(nx.connected_component_subgraphs(g), key=len, reverse=True)[0])
    #               for t, g in graphs]
    # else:
    #     giants = [(t, g.subgraph(sorted(nx.connected_components(g), key=len, reverse=True)[0]))
    #               for t, g in graphs]

    # return sorted(giants, key=lambda x: x[0])

    # if dataname == 'email-eucore':
    #     takerange = list(range(0, 17 + 1))
    # elif dataname == 'facebook-links':
    #     takerange = list(range(1, 27 + 1))
    # elif dataname == 'email-dnc':
    #     takerange = list(range(1, 7 + 1))
    # else:
    #     takerange = None

    return sorted(graphs, key=lambda x: x[0])
