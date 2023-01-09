import networkx as nx
import networkx.algorithms.isomorphism as iso

from src.utils import autodict, gamma, graph_mdl, graph_edit_distance
from cnrg.LightMultiGraph import convert
from cnrg.LightMultiGraph import LightMultiGraph as LMG


class MetaRule:
    '''
        Class attributes:
            rules = dict mapping a timestamp t to the version of this rule at that time
            edits = a dict mapping a pair of times (t1, t2) to the edit distance between those two versions of this rule
                    if a valid (t1, t2) timestamp pair is provided as a key with no corresponding value,
                        the edit distance will be automatically computed and inserted into the dict
            level = level of discovery during extraction (the root is at 0)
            idn = the index in the grammar's rule_list referring to this rule

        Class properties:
            times = a list containing the keys for the `rules` dict
            mdl = the description length of encoding the rule (in bits)
    '''
    __slots__ = ('rules', 'edits', 'level', 'idn')

    def __init__(self, rules: dict[int, 'Rule'] = None, idn: int = -1):
        self.rules: dict[int, Rule] = rules if rules else {}
        self.edits: dict[(int, int), int] = autodict(self.cost)
        self.level: int = 0
        self.idn: int = idn

    @property
    def times(self) -> list[int]:
        return list(self.rules.keys())

    @property
    def mdl(self) -> float:
        return sum(gamma(tt) + rr.mdl for tt, rr in self.rules.items())

    def ensure(self, t1: int, t2: int):
        assert t1 in self.times
        if t2 not in self.times:
            self.rules[t2] = self.rules[t1].copy()

    def cost(self, timepair: tuple[int, int]):
        t1, t2 = timepair
        if t2 not in self.times:
            raise AssertionError(f'!!! this rule does not exist at time {t2} !!!')
        if t1 not in self.times:  # TODO: decide on this design decision
            # return 0
            return self.rules[t2].graph.order() + self.rules[t2].graph.size()
        return graph_edit_distance(self.rules[t1].graph, self.rules[t2].graph)

    def compute_edits(self):
        self.edits = autodict(self.cost)

        times = self.times
        time_pairs = zip(times[:-1], times[1:])

        for t1, t2 in time_pairs:
            self.edits[(t1, t2)]  # pylint: disable=pointless-statement
            # self.edits[(t1, t2)] = graph_edit_distance(self.rules[t1].graph, self.rules[t2].graph)

    def copy(self) -> 'MetaRule':
        return MetaRule(rules=self.rules.copy(), idn=self.idn)

    def __iter__(self):
        yield from self.rules.values()

    def __getitem__(self, time: int) -> 'Rule':
        return self.rules[time]

    def __eq__(self, other):
        return NotImplemented

    def __str__(self):
        return '{' + ', '.join(f'{tt} @ {rr.lhs} -> {rr.graph.order()}' for tt, rr in self.rules.items()) + '}'

    def __repr__(self):
        return str(self)


class Rule:
    '''
        Class attributes:
            lhs = the left-hand side nonterminal symbol
            graph = the right-hand side graph
            alias = dict mapping ``original_node_names`` -> ``rule_node_names``
            frequency = the number of times this rule was extracted (up to isomorphism)
            subtree = the original names of the nodes this rule was extracted from
            idn = the index in the grammar's rule_list referring to this rule

        Class properties:
            nonterminals = list of nonterminal symbols on this rule's right-hand side
            mdl = the description length of encoding the rule (in bits)
    '''
    __slots__ = ('lhs', 'graph', 'alias', 'frequency', 'subtree', 'idn')

    def __init__(self, lhs: int, graph: nx.Graph, alias: dict[int, str] = None,
                 frequency: int = 1, subtree: set[int] = None, idn: int = -1):
        self.lhs: int = lhs
        self.graph: LMG = convert(graph)
        self.alias: dict[int, str] = alias if alias else {}
        self.frequency: int = frequency
        self.subtree: set[int] = subtree if subtree else set()
        self.idn: int = idn

    @property
    def nonterminals(self):
        return [d['label'] for _, d in self.graph.nodes(data=True) if 'label' in d]

    @property
    def mdl(self) -> float:
        b_deg = nx.get_node_attributes(self.graph, 'b_deg')
        max_boundary_degree = 0 if not b_deg else max(b_deg.values())
        return gamma(max(0, self.lhs)) + graph_mdl(self.graph) + gamma(self.frequency) + self.graph.order() * gamma(max_boundary_degree)

    def copy(self) -> 'Rule':
        return Rule(lhs=self.lhs,
                    graph=self.graph.copy(),
                    alias=self.alias.copy(),
                    frequency=self.frequency,
                    subtree=self.subtree.copy(),
                    idn=self.idn)

    def generalize_rhs(self):
        self.alias = {}
        internal_node_counter = 'a'

        for n in self.graph.nodes():
            self.alias[n] = internal_node_counter
            internal_node_counter = chr(ord(internal_node_counter) + 1)

        nx.relabel_nodes(self.graph, mapping=self.alias, copy=False)

    # equality based on attributed graph isomorphism
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

    # hashing based on the Weisfeiler-Lehman algorithm
    def __hash__(self):
        g = self.graph.copy()

        for _, d in g.nodes(data=True):
            if 'label' not in d:
                d['label'] = -1

        return int(nx.weisfeiler_lehman_graph_hash(g, node_attr='label'), 16)

    def __str__(self):
        st = f'{self.lhs} -> (n = {self.graph.order()}, m = {self.graph.size()})'
        if len(self.nonterminals) != 0:  # if it has non-terminals, print the sizes
            st += ' nt: {' + ','.join(map(str, self.nonterminals)) + '}'
        return st

    def __repr__(self):
        return str(self)
