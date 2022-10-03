from time import sleep

import networkx as nx
from tqdm import tqdm


class LightMultiGraph(nx.Graph):
    def __init__(self):
        nx.Graph.__init__(self)

    def size(self, weight=None):
        return int(super(LightMultiGraph, self).size(weight='weight'))

    def __repr__(self):
        return f'n = {self.order():_d} m = {self.size():_d}'

    def add_edge(self, u, v, attr_dict=None, **attr):
        # print(f'inside add_edge {u}, {v}')
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
            # print(f'edge ({u}, {v}) exists, {self[u][v]["weight"]}')
            self[u][v]['weight'] += wt
            if edge_colors is not None:
                self[u][v]['edge_colors'] += edge_colors
            if edge_color is not None:
                self[u][v]['edge_color'] = edge_color
            if attr_records is not None:
                self[u][v]["attr_records"] += attr_records
        else:
            if edge_colors is not None:
                super(LightMultiGraph, self).add_edge(u, v, weight=wt, edge_colors=edge_colors)
            elif edge_color is not None:
                super(LightMultiGraph, self).add_edge(u, v, weight=wt, edge_color=edge_color)
            elif attr_records is not None:
                super(LightMultiGraph, self).add_edge(
                    u, v, weight=wt, attr_records=attr_records
                )
            else:
                super(LightMultiGraph, self).add_edge(u, v, weight=wt)

    def copy(self):
        legacy = False
        if legacy:
            g_copy = LightMultiGraph()

            for node, d in self.nodes(data=True):
                if len(d) == 0:  # prevents adding an empty 'attr_dict' dictionary
                    g_copy.add_node(node)
                else:
                    if 'attr_records' in d:  # attr_records supercedes other attributes (excluding edge weight)
                        g_copy.add_node(node, attr_records=d['attr_records'])
                    elif 'b_deg' in d and 'label' in d and 'node_colors' in d:  # this keeps the label and the b_deg attributes to the same level
                        g_copy.add_node(node, b_deg=d['b_deg'], label=d['label'], node_colors=d['node_colors'])
                    elif 'b_deg' in d and 'label' in d:  # this keeps the label and the b_deg attributes to the same level
                        g_copy.add_node(node, b_deg=d['b_deg'], label=d['label'])
                    elif 'b_deg' in d and 'node_colors' in d:  # this keeps the label and the b_deg attributes to the same level
                        g_copy.add_node(node, b_deg=d['b_deg'], node_colors=d['node_colors'])
                    elif 'label' in d and 'node_colors' in d:  # this keeps the label and the b_deg attributes to the same level
                        g_copy.add_node(node, label=d['label'], node_colors=d['node_colors'])
                    elif 'b_deg' in d:
                        g_copy.add_node(node, b_deg=d['b_deg'])
                    elif 'label' in d:
                        g_copy.add_node(node, label=d['label'])
                    elif 'node_colors' in d:
                        g_copy.add_node(node, node_colors=d['node_colors'])
                    if 'appears' in d:
                        g_copy.nodes[node]['appears'] = d['appears']

            for e in self.edges(data=True):
                u, v, d = e
                g_copy.add_edge(u, v, attr_dict=d)

            return g_copy
        else:
            return super(LightMultiGraph, self).copy()

    def add_edges_from(self, ebunch, attr_dict=None, **attr):
        for e in ebunch:
            ne = len(e)
            if ne == 3:
                u, v, dd = e
            elif ne == 2:
                u, v = e
                dd = {}  # doesnt need edge_attr_dict_factory
            else:
                raise nx.NetworkXError("Edge tuple %s must be a 2-tuple or 3-tuple." % (e,))
            if attr_dict is not None:
                self.add_edge(u, v, attr_dict={**dd, **attr_dict}, **attr)
            else:
                self.add_edge(u, v, attr_dict=dd, **attr)

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
