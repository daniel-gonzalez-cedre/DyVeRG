from typing import Any

import networkx as nx
import networkx.algorithms.isomorphism as iso

# import cnrg.MDL as MDL
from cnrg import MDL
from cnrg.LightMultiGraph import convert
from cnrg.LightMultiGraph import LightMultiGraph as LMG


class BaseRule:
    '''
        Prototypical rule in a vertex-replacement graph grammar

        Class attributes:
            lhs = the left-hand side nonterminal symbol
            graph = the right-hand side graph
            level = level of discovery during extraction (the root is at 0)
            frequency = the number of times this rule was extracted (up to isomorphism)
            idn = the index in the grammar's rule_list referring to this rule
            time = the timestamp when this rule was extracted
            time_changed = the most recent timestamp when this rule was modified
            subtree = the original names of the nodes this rule was extracted from
            alias = dict mapping ``original_node_names`` -> ``rule_node_names``
            edit_dist = the minimum graph edit distance between this rule and all other rules in the grammar
                        == 0 if this rule was added during the course of learning a static grammar
                        >= 0 if this rule was added by merging two grammars or modifying a previous rule
            timed_out = a flag indicating whether or not the edit_dist computation timed out at any point for this rule

        Class properties:
            mdl = the description length of encoding the rule (in bits)
            dl = the description length of encoding the rule (in bits)
            nonterminals = list of nonterminal symbols on this rule's right-hand side
    '''

    __slots__ = (
        'lhs', 'graph', 'level', 'frequency', 'idn',
        'subtree', 'alias', 'time_created', 'time_changed',
        'edit_dist', 'timed_out'
    )

    def __init__(self, lhs: int, graph: nx.Graph, level: int = 0, frequency: int = 1, idn: int = None,
                 time: int = None, subtree: set[int] = None, alias: dict[int, Any] = None,
                 edit_dist: int = 0, timed_out: bool = False):
        self.lhs: int = lhs
        self.graph: LMG = convert(graph)
        self.level: int = level
        self.frequency: int = frequency
        self.idn: int = idn
        self.time_created: int = time
        self.time_changed: int = time
        self.subtree: set[int] = subtree if subtree is not None else set()
        self.alias: dict[int, str] = alias if alias is not None else {}
        self.edit_dist: int = edit_dist
        self.timed_out: bool = timed_out

    @property
    def nonterminals(self):
        return [d['label'] for _, d in self.graph.nodes(data=True) if 'label' in d]

    @property
    def mdl(self) -> float:
        return self.dl

    @property
    def dl(self) -> float:
        raise NotImplementedError

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

    def __str__(self):
        st = f'{self.lhs} -> (n = {self.graph.order()}, m = {self.graph.size()})'
        if len(self.nonterminals) != 0:  # if it has non-terminals, print the sizes
            st += ' nt: {' + ','.join(map(str, self.nonterminals)) + '}'
        if self.frequency > 1:  # if freq > 1, show it in square brackets
            st += f'[{self.frequency}]'
        return st

    def __repr__(self):
        st = f'{self.lhs} -> ({self.graph.order()}, {self.graph.size()})'
        if len(self.nonterminals) != 0:  # if it has non-terminals, print the sizes
            st += '{' + ','.join(map(str, self.nonterminals)) + '}'
        if self.frequency > 1:  # if freq > 1, show it in square brackets
            st += f'[{self.frequency}]'
        return st

    # isomorphism-based equality checking
    def __eq__(self, other):  # two rules are equal if the LHSs match and RHSs are isomorphic
        g1 = nx.convert_node_labels_to_integers(self.graph)
        g2 = nx.convert_node_labels_to_integers(other.graph)
        # and nx.fast_could_be_isomorphic(g1, g2) \
        return self.lhs == other.lhs and g1.order() == g2.order() and g1.size() == g2.size() and nx.is_isomorphic(
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

    def copy(self):
        copy_rule = BaseRule(
            lhs=self.lhs,
            graph=self.graph.copy(),
            level=self.level,
            frequency=self.frequency,
            idn=self.idn,
            time=self.time_changed,
            subtree=self.subtree.copy(),
            alias=self.alias.copy(),
            edit_dist=self.edit_dist,
            timed_out=self.timed_out
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


class PartRule(BaseRule):
    """
    Rule class for Partial option
    """
    def __init__(self, lhs: int, graph: nx.Graph, level: int = 0, frequency: int = 1, idn: int = None,
                 time: int = None, subtree: set[int] = None, alias: dict[int, str] = None,
                 edit_dist: int = 0, timed_out: bool = False):
        super().__init__(
            lhs=lhs,
            graph=graph,
            level=level,
            frequency=frequency,
            idn=idn,
            time=time,
            subtree=subtree,
            alias=alias,
            edit_dist=edit_dist,
            timed_out=timed_out
        )

    @property
    def dl(self) -> float:
        """
        Calculates the MDL for the rule. This includes the encoding of boundary degrees of the nodes.
        l_u = 2 (because we have one type of nodes and one type of edge)
        :return:
        """
        b_deg = nx.get_node_attributes(self.graph, 'b_deg')
        assert len(b_deg) > 0, 'invalid b_deg'
        max_boundary_degree = max(b_deg.values())
        # l_u = 2
        # for node, data in self.graph.nodes(data=True):
        #     if 'label' in data:  # it's a non-terminal
        #         l_u = 3
        return MDL.gamma_code(max(0, self.lhs) + 1) + MDL.graph_dl(self.graph) + MDL.gamma_code(self.frequency + 1) + self.graph.order() * MDL.gamma_code(max_boundary_degree + 1)

    def copy(self):
        copy_rule = PartRule(
            lhs=self.lhs,
            graph=self.graph.copy(),
            level=self.level,
            frequency=self.frequency,
            idn=self.idn,
            time=self.time_changed,
            subtree=self.subtree.copy(),
            alias=self.alias.copy(),
            edit_dist=self.edit_dist,
            timed_out=self.timed_out
        )
        copy_rule.time_changed = self.time_changed
        return copy_rule

    def generalize_rhs(self):
        """
        Relabels the RHS such that the internal nodes are Latin characters, the boundary nodes are numerals.

        :param self: RHS subgraph
        :return:
        """
        self.alias = {}
        internal_node_counter = 'a'

        for n in self.graph.nodes():
            self.alias[n] = internal_node_counter
            internal_node_counter = chr(ord(internal_node_counter) + 1)

        nx.relabel_nodes(self.graph, mapping=self.alias, copy=False)


class FullRule(BaseRule):
    """
    Rule object for full-info option
    """
    __slots__ = 'internal_nodes', 'edges_covered'

    def __init__(self, lhs, graph, internal_nodes, level=0, frequency=1, edges_covered=None, idn=None, time=None, edit_dist=0, timed_out: bool = False):
        super().__init__(
            lhs=lhs,
            graph=graph,
            level=level,
            frequency=frequency,
            time=time,
            edit_dist=edit_dist,
            timed_out=timed_out
        )
        self.internal_nodes = internal_nodes  # the set of internal nodes
        self.edges_covered = edges_covered  # edges in the original graph that's covered by the rule

    @property
    def dl(self) -> float:
        """
        Updates the MDL cost of the RHS. l_u is the number of unique entities in the graph.
        We have two types of nodes (internal and external) and one type of edge
        :return:
        """
        return MDL.gamma_code(max(0, self.lhs) + 1) + MDL.graph_dl(self.graph) + MDL.gamma_code(self.frequency + 1)

    def copy(self):
        copy_rule = FullRule(
            lhs=self.lhs,
            graph=self.graph.copy(),
            level=self.level,
            frequency=self.frequency,
            internal_nodes=self.internal_nodes.copy(),
            edges_covered=self.edges_covered.copy(),
            idn=self.idn,
            time=self.time_changed,
            edit_dist=self.edit_dist,
            timed_out=self.timed_out
        )
        copy_rule.time_changed = self.time_changed
        return copy_rule

    def generalize_rhs(self):
        """
        Relabels the RHS such that the internal nodes are Latin characters, the boundary nodes are numerals.

        :param self: RHS subgraph
        :return:
        """
        self.alias = {}
        internal_node_counter = 'a'
        boundary_node_counter = 0

        for n in self.internal_nodes:
            self.alias[n] = internal_node_counter
            internal_node_counter = chr(ord(internal_node_counter) + 1)

        for n in [x for x in self.graph.nodes() if x not in self.internal_nodes]:
            self.alias[n] = boundary_node_counter
            boundary_node_counter += 1

        self.graph = nx.relabel_nodes(self.graph, mapping=self.alias)
        self.internal_nodes = {self.alias[n] for n in self.internal_nodes}

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


class NoRule(PartRule):
    """
    Class for no_info
    """

    def copy(self):
        copy_rule = NoRule(
            lhs=self.lhs,
            graph=self.graph.copy(),
            level=self.level,
            frequency=self.frequency,
            idn=self.idn,
            time=self.time_changed,
            edit_dist=self.edit_dist,
            timed_out=self.timed_out
        )
        copy_rule.time_changed = self.time_changed
        return copy_rule

    @property
    def dl(self) -> float:
        """
        Calculates the MDL for the rule. This just includes encoding the graph.
        l_u = 2 (because we have one type of nodes and one type of edge)
        :return:
        """
        return MDL.gamma_code(max(0, self.lhs) + 1) + MDL.graph_dl(self.graph) + MDL.gamma_code(self.frequency + 1)
