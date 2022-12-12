from time import sleep
from typing import Union

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
            graphcopy.add_edge(u, v, **d)

        return graphcopy

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        u, v = (u_of_edge, v_of_edge)
        attr: dict = attr if attr else {}
        weight: int = attr.get('weight', 1)
        colors: Union[list, set, int, float, str] = attr.get('colors')
        if isinstance(colors, (list, set)):
            colors = list(colors)
        elif isinstance(colors, (int, float, str)):
            colors = [colors]

        if self.has_edge(u, v):  # edge already exists
            self[u][v]['weight'] += weight
            if colors and 'colors' in self[u][v]:
                self[u][v]['colors'] += colors
            elif colors and 'colors' not in self[u][v]:
                self[u][v]['colors'] = colors
        else:
            new_attr = {'weight': weight}
            if colors:
                new_attr['colors'] = colors
            super().add_edge(u, v, **new_attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        for e in ebunch_to_add:
            ne = len(e)

            if ne == 3:
                u, v, d = e
            elif ne == 2:
                u, v = e
                d = {}
            else:
                raise nx.NetworkXError(f'Edge tuple {e} must be a 2-tuple or 3-tuple.')

            self.add_edge(u, v, **d)

    def remove_edge(self, u, v):
        d = self.edges[u, v]
        d['weight'] -= 1
        if d['weight'] <= 0:
            super().remove_edge(u, v)

    def number_of_edges(self, u=None, v=None):
        if u is None and v is None:
            return self.size()  # number of edges in the graph
        if u is None or v is None:
            return 0  # nonsense
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
