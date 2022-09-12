import random
from os.path import join
from typing import List, Tuple

import networkx as nx
from tqdm import tqdm

# sys.path.append('/g/g15/cedre/pysparkplug')
# from pysp.utils.optsutil import get_inv_map, map_to_integers


def read_data(dataname: str = 'nips', lookback: int = 10, num_epochs: int = 10) -> Tuple[List[nx.Graph], List[int]]:
    rootpath = '/Users/danielgonzalez/repos/LLNL/'

    if dataname == 'fb-messages' or dataname == 'email-dnc':
        datapath = f'gonzalez_mrhyde/data/{dataname}/{dataname}.edges'
        graph = nx.read_edgelist(join(rootpath, datapath),
                                 delimiter=',',
                                 nodetype=int,
                                 data=[('time', float)])

        edges = sorted([(u, v, d['time']) for u, v, d in graph.edges(data=True)],
                       key=lambda x: x[2])
        times = sorted([t for u, v, t in edges])
        delta = len(times) // num_epochs
        edge_brackets = [edges[k * delta:(k + 1) * delta] for k in range(num_epochs + 1)]
        edge_brackets[num_epochs - 1] += edge_brackets[num_epochs]
        del edge_brackets[num_epochs]

        brackets_stripped = [[(u, v) for u, v, t in bracket]
                             for bracket in edge_brackets]
        graphs = [graph.edge_subgraph(stripped)
                  for stripped in brackets_stripped]

        if True:
            if float(nx.__version__[:3]) < 2.4:
                graphs = [sorted(nx.connected_component_subgraphs(g), key=len, reverse=True)[0]
                          for g in graphs]
            else:
                graphs = [g.subgraph(sorted(nx.connected_components(g), key=len, reverse=True)[0])
                          for g in graphs]

        if lookback > 0:
            cum_graphs = []

            for i in range(len(graphs)):
                cum_nodes = set()
                cum_edges = set()

                for g in graphs[max(0, i - lookback + 1):i + 1]:
                    cum_nodes |= set(g.nodes())
                    cum_edges |= set(g.edges())

                cum_graph = nx.Graph()
                cum_graph.add_nodes_from(cum_nodes)
                cum_graph.add_edges_from(cum_edges)

                cum_graphs.append(cum_graph)

            assert len(cum_graphs) == len(graphs)

        else:
            cum_graphs = graphs

        if float(nx.__version__[:3]) < 2.4:
            giants = [sorted(nx.connected_component_subgraphs(cum_graph),
                             key=len,
                             reverse=True)[0]
                      for g in cum_graphs]
        else:
            giants = [g.subgraph(sorted(nx.connected_components(g),
                                        key=len,
                                        reverse=True)[0])
                      for g in cum_graphs]

        years = list(range(num_epochs))

    elif dataname == 'ca-cit-HepPh' or dataname == 'ca-cit-HepTh' or dataname == 'tech-as-topology':
        datapath = f'gonzalez_mrhyde/data/{dataname}/{dataname}.edges'
        graph = nx.read_edgelist(join(rootpath, datapath),
                                 delimiter=' ',
                                 nodetype=int,
                                 data=[('', int), ('time', int)])

        edges = sorted([(u, v, d['time']) for u, v, d in graph.edges(data=True)],
                       key=lambda x: x[2])
        times = sorted([t for u, v, t in edges])
        delta = len(times) // num_epochs

        edge_brackets = [edges[k * delta:(k + 1) * delta]
                         for k in range(num_epochs + 1)]
        edge_brackets[num_epochs - 1] += edge_brackets[num_epochs]
        del edge_brackets[num_epochs]

        edges_clean = [[(u, v) for u, v, t in bracket]
                       for bracket in edge_brackets]

        graphs = [graph.edge_subgraph(clean)
                  for clean in edges_clean]

        if lookback > 0:
            cum_graphs = []

            for i in range(len(graphs)):
                cum_nodes = set()
                cum_edges = set()

                for g in graphs[max(0, i - lookback + 1):i + 1]:
                    cum_nodes |= set(g.nodes())
                    cum_edges |= set(g.edges())

                cum_graph = nx.Graph()
                cum_graph.add_nodes_from(cum_nodes)
                cum_graph.add_edges_from(cum_edges)

                cum_graphs.append(cum_graph)

            assert len(cum_graphs) == len(graphs)

        else:
            cum_graphs = graphs

        if float(nx.__version__[:3]) < 2.4:
            giants = [sorted(nx.connected_component_subgraphs(cum_graph),
                             key=len,
                             reverse=True)[0]
                      for g in cum_graphs]
        else:
            giants = [g.subgraph(sorted(nx.connected_components(g),
                                        key=len,
                                        reverse=True)[0])
                      for g in cum_graphs]

        years = list(range(num_epochs))

    elif dataname == '':
        raise NotImplementedError

    elif dataname == 'clique-ring':
        # DO NOT USE THIS DATASET, IT IS CAUSING INFINITE RECURSION
        init_graph = nx.read_edgelist('/g/g15/cedre/gonzalez_mrhyde/data/clique-ring-500-4.g')
        edges = []

        for itr in range(6):
            start = (5 * itr) + 1999
            edges += [(start, start + 1), (start + 1, start + 2),
                      (start + 2, start + 3), (start + 3, start + 4),
                      (start + 4, start), (start, start + 2),
                      (start + 1, start + 3), (start + 2, start + 4),
                      (start, start + 3), (start + 1, start + 4), (start + 4, 0)]

        next_graph = init_graph.copy()
        next_graph.add_edges_from(edges)

        giants = [init_graph, next_graph]
        years = [0, 1]

    return giants, years

    # if cumulative:
    #     cum_giants = [giants[0]]

    #     for idx, giant in enumerate(giants[1:]):
    #         next_graph = cum_giants[idx].copy()
    #         next_graph.add_nodes_from(giant.nodes())
    #         next_graph.add_edges_from(giant.edges())
    #         next_giant = next_graph.subgraph(sorted(nx.connected_components(next_graph), key=len, reverse=True)[0])
    #         cum_giants.append(next_giant)

    #     return cum_giants, list(range(num_epochs))

    # else:
    #     return giants, years


def create_graphs(author_series, years, cumulative: bool = True) -> List[nx.Graph]:
    graphs = [nx.Graph() for _ in years]

    # author_series[i][year - 1987] is the list of coauthors for author i in the given year
    for author, author_list in enumerate(author_series):
        assert len(author_list) == len(years)

        for year, coauthors in enumerate(author_list):
            graphs[year].add_edges_from([(author, c) for c in coauthors])

    if cumulative:
        # make the NeurIPS dataset cumulative
        # ensure the graph is connected at each time step BY RANDOMLY ADDING EDGES (!!)
        #   ^^^ this is just for testing ^^^
        for year, graph in tqdm(enumerate(graphs)):
            if year > 0:
                graph.add_nodes_from(graphs[year - 1].nodes())
                graph.add_edges_from(graphs[year - 1].edges())

            while not nx.is_connected(graph):
                component1, component2 = random.sample(list(nx.connected_components(graph)), 2)
                u, = random.sample(component1, 1)
                v, = random.sample(component2, 1)
                graph.add_edge(u, v)

        for graph in graphs:
            assert nx.is_connected(graph)
    else:
        for year, graph in enumerate(graphs):
            graphs[year] = max(nx.connected_component_subgraphs(graph), key=len)

    return graphs
