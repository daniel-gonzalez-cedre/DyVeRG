from time import sleep

import networkx as nx
from tqdm import tqdm


def convert(g: nx.Graph):
    if isinstance(g, LightMultiGraph):
        return g
    g_lmg = LightMultiGraph()
    g_lmg.add_nodes_from(g.nodes(data=True))
    g_lmg.add_edges_from(g.edges(data=True))
    return g_lmg


class LightMultiGraph(nx.Graph):
    def __init__(self):
        nx.Graph.__init__(self)

    def size(self, weight=None):
        return int(super().size(weight='weight'))

    def __repr__(self):
        return f'n = {self.order():_d} m = {self.size():_d}'

    # do NOT call super().copy(); see line 64
    def copy(self, as_view=False):
        graphcopy = LightMultiGraph()
        graphcopy.add_nodes_from(self.nodes(data=True))

        for u, v, d in self.edges(data=True):
            graphcopy.add_edge(u, v, attr_dict=d)

        return graphcopy

    def add_edge(self, u_of_edge, v_of_edge, attr_dict=None, **attr):
        u, v = (u_of_edge, v_of_edge)
        edge_colors = None
        edge_color = None
        attr_records = None

        if attr_dict is not None and 'weight' in attr_dict:
            wt = attr_dict['weight']
        elif attr is not None and 'weight' in attr:
            wt = attr['weight']
        else:
            wt = 1

        if attr_dict is not None and 'attr_records' in attr_dict:
            attr_records = attr_dict['attr_records']
        elif attr is not None and 'attr_records' in attr:
            attr_records = attr['attr_records']

        if attr_dict is not None and 'edge_colors' in attr_dict:
            edge_colors = attr_dict['edge_colors']
        elif attr is not None and 'edge_colors' in attr:
            edge_colors = attr['edge_colors']
        elif attr_dict is not None and 'edge_color' in attr_dict:
            edge_color = attr_dict['edge_color']
        elif attr is not None and 'edge_color' in attr:
            edge_color = attr['edge_color']

        if self.has_edge(u, v):  # edge already exists
            self[u][v]['weight'] += wt
            if edge_colors is not None:
                self[u][v]['edge_colors'] += edge_colors
            if edge_color is not None:
                self[u][v]['edge_color'] = edge_color
            if attr_records is not None:
                self[u][v]["attr_records"] += attr_records
        else:
            if edge_colors is not None:
                super().add_edge(u, v, weight=wt, edge_colors=edge_colors)
            elif edge_color is not None:
                super().add_edge(u, v, weight=wt, edge_color=edge_color)
            elif attr_records is not None:
                super().add_edge(
                    u, v, weight=wt, attr_records=attr_records
                )
            else:
                super().add_edge(u, v, weight=wt)

    def add_edges_from(self, ebunch_to_add, attr_dict=None, **attr):
        for e in ebunch_to_add:
            ne = len(e)
            if ne == 3:
                u, v, dd = e
            elif ne == 2:
                u, v = e
                dd = {}  # doesnt need edge_attr_dict_factory
            else:
                raise nx.NetworkXError('Edge tuple %s must be a 2-tuple or 3-tuple.' % (e,))
            if attr_dict is not None:
                self.add_edge(u, v, attr_dict={**dd, **attr_dict}, **attr)
            else:
                self.add_edge(u, v, attr_dict=dd, **attr)

    def remove_edge(self, u, v):
        d = self.edges[u, v]
        d['weight'] -= 1
        if d['weight'] <= 0:
            super().remove_edge(u, v)

    def number_of_edges(self, u=None, v=None):
        if u is None:
            return self.size()
        try:
            return self[u][v]['weight']
        except KeyError:
            return 0  # no such edge


if __name__ == '__main__':
    # g = RHSGraph()
    # g.add_edge(1, 2)
    # g.add_edge(2, 3)
    # g.add_edge(1, 2)
    #
    # print(g.edges(data=True))
    # print(g.number_of_edges(1, 2))
    with tqdm(total=100) as pbar:
        perc = 0
        while perc <= 100:
            sleep(0.1)
            # pbar.update(pbar.n - perc)
            pbar.update(perc - pbar.n)
            perc += 5
