# import numpy as np
import networkx as nx
import networkx.algorithms.isomorphism as iso

# import cnrg.MDL as MDL
from cnrg import MDL
from cnrg.LightMultiGraph import convert


class BaseRule:
    """
    Base class for Rule
    """

    __slots__ = "lhs", "graph", "level", "cost", "frequency", "id", "non_terminals", "subtree", "time", "time_changed", "edit_cost"

    def __init__(self, lhs, graph, level=0, cost=0, frequency=1, id=None, time=None, edit_cost=0):
        self.lhs = lhs  # the left hand side: the number of boundary edges
        self.graph = convert(graph)  # the right hand side subgraph
        self.level = level  # level of discovery in the tree (the root is at 0)
        self.cost = cost  # the cost of encoding the rule using MDL (in bits)
        self.frequency = frequency  # frequency of occurrence
        self.id = id
        self.time = time
        self.time_changed = time
        self.non_terminals = [d['label'] for _, d in self.graph.nodes(data=True) if 'label' in d]
        # store original vertex ids (subgraph-to-subgraph dynamics)
        self.subtree = None
        # for _, d in self.graph.nodes(data=True):
        #     if 'label' in d:
        #         self.non_terminals.append(d['label'])
        self.edit_cost: int = edit_cost

    # approximate equality using Weisfeiler-Lehman graph hashing
    def hash_equals(self, other):
        if self.lhs != other.lhs:
            return False

        g1 = self.graph.copy()
        g2 = other.graph.copy()

        for _, d in g1.nodes(data=True):
            if 'label' not in d:
                d['label'] = -1

        for _, d in g2.nodes(data=True):
            if 'label' not in d:
                d['label'] = -1

        hash1 = nx.weisfeiler_lehman_graph_hash(g1, node_attr='label')
        hash2 = nx.weisfeiler_lehman_graph_hash(g2, node_attr='label')

        return hash1 == hash2

        # return (self.lhs == other.lhs) and (nx.weisfeiler_lehman_graph_hash(self.graph) == nx.weisfeiler_lehman_graph_hash(other.graph))

    def __str__(self):
        st = '({}) {} -> (n = {}, m = {})'.format(self.id, self.lhs, self.graph.order(), self.graph.size())
        # print non-terminals if present

        if len(self.non_terminals) != 0:  # if it has non-terminals, print the sizes
            st += 'nt: {' + ','.join(map(str, self.non_terminals)) + '}'

        if self.frequency > 1:  # if freq > 1, show it in square brackets
            st += '[{}]'.format(self.frequency)
        return st

    def __repr__(self):
        st = '{} -> ({}, {})'.format(self.lhs, self.graph.order(), self.graph.size())

        if len(self.non_terminals) != 0:  # if it has non-terminals, print the sizes
            st += '{' + ','.join(map(str, self.non_terminals)) + '}'
        if self.frequency > 1:  # if freq > 1, show it in square brackets
            st += '[{}]'.format(self.frequency)
        return st

    # isomorphism-based equality checking
    def __eq__(self, other):  # two rules are equal if the LHSs match and RHSs are isomorphic
        g1 = nx.convert_node_labels_to_integers(self.graph)
        g2 = nx.convert_node_labels_to_integers(other.graph)
        # and nx.fast_could_be_isomorphic(g1, g2) \
        return self.lhs == other.lhs and nx.is_isomorphic(
            g1,
            g2,
            edge_match=iso.numerical_edge_match('weight', 1.0),  # pylint: disable=not-callable
            node_match=iso.categorical_node_match('label', '')
        )

    def __hash__(self):
        g = self.graph.copy()

        for _, d in g.nodes(data=True):
            if 'label' not in d:
                d['label'] = -1

        return int(nx.weisfeiler_lehman_graph_hash(g, node_attr='label'), 16)

        # g = nx.freeze(self.graph)
        # return hash((self.lhs, g))

    def __deepcopy__(self, memodict={}):
        copy_rule = BaseRule(
            lhs=self.lhs,
            graph=self.graph.copy(),
            level=self.level,
            cost=self.cost,
            frequency=self.frequency,
            id=self.id,
            time=self.time,
            edit_cost=self.edit_cost
        )
        copy_rule.time_changed = self.time_changed
        return copy_rule

    def contract_rhs(self):
        pass

    def draw(self):
        '''
        Returns a graphviz object that can be rendered into a pdf/png
        '''
        from graphviz import Graph
        flattened_graph = nx.Graph(self.graph)

        dot = Graph(engine='dot')
        for node, d in self.graph.nodes(data=True):
            if 'label' in d:
                dot.node(str(node), str(d['label']), shape='square', height='0.20')
            else:
                dot.node(str(node), '', height='0.12', shape='circle')

        for u, v in flattened_graph.edges():
            w = self.graph.number_of_edges(u, v)
            if w > 1:
                dot.edge(str(u), str(v), label=str(w))
            else:
                dot.edge(str(u), str(v))
        return dot

    def deactivate(self):
        self.is_active = False

    def activate(self):
        self.is_active = True


class FullRule(BaseRule):
    """
    Rule object for full-info option
    """
    __slots__ = 'internal_nodes', 'edges_covered'

    def __init__(
        self,
        lhs,
        graph,
        internal_nodes,
        level=0,
        cost=0,
        frequency=1,
        edges_covered=None,
        id=None,
        time=None,
        edit_cost=0
    ):
        super().__init__(lhs=lhs, graph=graph, level=level, cost=cost, frequency=frequency, time=time, edit_cost=edit_cost)
        self.internal_nodes = internal_nodes  # the set of internal nodes
        self.edges_covered = edges_covered  # edges in the original graph that's covered by the rule
        self.subtree = None

    def __deepcopy__(self, memodict={}):
        copy_rule = FullRule(
            lhs=self.lhs,
            graph=self.graph.copy(),
            level=self.level,
            cost=self.cost,
            frequency=self.frequency,
            internal_nodes=self.internal_nodes,
            edges_covered=self.edges_covered,
            id=self.id,
            time=self.time,
            edit_cost=self.edit_cost
        )
        copy_rule.time_changed = self.time_changed
        return copy_rule

    def copy(self):
        copy_rule = FullRule(
            lhs=self.lhs,
            graph=self.graph.copy(),
            level=self.level,
            cost=self.cost,
            frequency=self.frequency,
            internal_nodes=self.internal_nodes,
            edges_covered=self.edges_covered,
            id=self.id,
            time=self.time,
            edit_cost=self.edit_cost
        )
        copy_rule.time_changed = self.time_changed
        return copy_rule

    def calculate_cost(self):
        """
        Updates the MDL cost of the RHS. l_u is the number of unique entities in the graph.
        We have two types of nodes (internal and external) and one type of edge
        :return:
        """
        self.cost = (
            MDL.gamma_code(max(0, self.lhs) + 1) +
            MDL.graph_dl(self.graph) +
            MDL.gamma_code(self.frequency + 1)
        )

    def generalize_rhs(self):
        """
        Relabels the RHS such that the internal nodes are Latin characters, the boundary nodes are numerals.

        :param self: RHS subgraph
        :return:
        """
        mapping = {}
        internal_node_counter = 'a'
        boundary_node_counter = 0

        for n in self.internal_nodes:
            mapping[n] = internal_node_counter
            internal_node_counter = chr(ord(internal_node_counter) + 1)

        for n in [x for x in self.graph.nodes() if x not in self.internal_nodes]:
            mapping[n] = boundary_node_counter
            boundary_node_counter += 1
        self.graph = nx.relabel_nodes(self.graph, mapping=mapping)
        self.internal_nodes = {mapping[n] for n in self.internal_nodes}

    def contract_rhs(self):
        """
        Contracts the RHS such that all boundary nodes with degree 1 are replaced by a special boundary isolated node I
        """
        iso_nodes = set()
        for node, dd in self.graph.nodes(data=True):
            if node not in self.internal_nodes and self.graph.degree(node) == 1:  # identifying the isolated nodes
                iso_nodes.add((node, dd))

        if len(iso_nodes) == 0:  # the rule cannot be contracted
            self.generalize_rhs()
            return

        rhs_copy = nx.Graph(self.graph)

        for iso_node in iso_nodes:
            for u in rhs_copy.neighbors(iso_node):
                self.graph.add_edge(u, 'Iso', attr_dict={'b': True})

        assert self.graph.has_node('Iso'), 'No Iso node after contractions'

        self.graph.remove_nodes_from(iso_nodes)  # remove the old isolated nodes

        self.generalize_rhs()
        return


class PartRule(BaseRule):
    """
    Rule class for Partial option
    """
    def __init__(self, lhs, graph, level=0, cost=0, frequency=1, id=None, time=None, edit_cost=0):
        super().__init__(
            lhs=lhs,
            graph=graph,
            level=level,
            cost=cost,
            frequency=frequency,
            id=id,
            time=time,
            edit_cost=edit_cost
        )
        self.subtree = None

    def __deepcopy__(self, memodict={}):
        copy_rule = PartRule(
            lhs=self.lhs,
            graph=self.graph.copy(),
            level=self.level,
            cost=self.cost,
            frequency=self.frequency,
            id=self.id,
            time=self.time,
            edit_cost=self.edit_cost
        )
        copy_rule.time_changed = self.time_changed
        return copy_rule

    def copy(self):
        copy_rule = PartRule(
            lhs=self.lhs,
            graph=self.graph.copy(),
            level=self.level,
            cost=self.cost,
            frequency=self.frequency,
            id=self.id,
            time=self.time,
            edit_cost=self.edit_cost
        )
        copy_rule.time_changed = self.time_changed
        return copy_rule

    def generalize_rhs(self):
        """
        Relabels the RHS such that the internal nodes are Latin characters, the boundary nodes are numerals.

        :param self: RHS subgraph
        :return:
        """
        mapping = {}
        internal_node_counter = 'a'

        for n in self.graph.nodes():
            mapping[n] = internal_node_counter
            internal_node_counter = chr(ord(internal_node_counter) + 1)

        nx.relabel_nodes(self.graph, mapping=mapping, copy=False)

    def calculate_cost(self):
        '''
        Calculates the MDL for the rule. This includes the encoding of boundary degrees of the nodes.
        l_u = 2 (because we have one type of nodes and one type of edge)
        :return:
        '''
        b_deg = nx.get_node_attributes(self.graph, 'b_deg')
        assert len(b_deg) > 0, 'invalid b_deg'
        max_boundary_degree = max(b_deg.values())
        # l_u = 2
        # for node, data in self.graph.nodes(data=True):
        #     if 'label' in data:  # it's a non-terminal
        #         l_u = 3
        self.cost = (
            MDL.gamma_code(max(0, self.lhs) + 1) +
            MDL.graph_dl(self.graph) +
            MDL.gamma_code(self.frequency + 1) +
            self.graph.order() * MDL.gamma_code(max_boundary_degree + 1)
        )


class NoRule(PartRule):
    """
    Class for no_info
    """
    def __deepcopy__(self, memodict={}):
        copy_rule = NoRule(
            lhs=self.lhs,
            graph=self.graph.copy(),
            level=self.level,
            cost=self.cost,
            frequency=self.frequency,
            id=self.id,
            time=self.time,
            edit_cost=self.edit_cost
        )
        copy_rule.time_changed = self.time_changed
        return copy_rule

    def copy(self, memodict={}):
        copy_rule = NoRule(
            lhs=self.lhs,
            graph=self.graph.copy(),
            level=self.level,
            cost=self.cost,
            frequency=self.frequency,
            id=self.id,
            time=self.time,
            edit_cost=self.edit_cost
        )
        copy_rule.time_changed = self.time_changed
        return copy_rule

    def calculate_cost(self):
        """
        Calculates the MDL for the rule. This just includes encoding the graph.
        l_u = 2 (because we have one type of nodes and one type of edge)
        :return:
        """
        self.cost = (
            MDL.gamma_code(max(0, self.lhs) + 1)
            + MDL.graph_dl(self.graph)
            + MDL.gamma_code(self.frequency + 1)
        )
