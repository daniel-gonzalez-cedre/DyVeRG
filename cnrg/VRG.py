"""
refactored VRG
"""
from typing import Union, Collection

from joblib import Parallel, delayed
import numpy as np
import networkx.algorithms.isomorphism as iso

from cnrg.Rule import BaseRule
from src.utils import find, replace, graph_edit_distance


class VRG:
    """
        Vertex-replacement graph grammar

        Class attributes:
            gtype = type of grammar
            clustering = node clustering algorithm used on the input graph to extract this grammar
            name = a string naming the graph this grammar was extracted from
            mu = the μ hyperparameter that (indirectly) determines the rules' maximum RHS sizes
            extraction_map = maps the dendrogram nodes to the rules extracted from them
                             used during the extraction process to compute self.decomposition

            decomposition = extraction tree for the grammar (a.k.a. decomposition for the input graph)
                            each entry looks like [rule, pidx, anode], where:
                                rule = the rule extracted at this point in the decomposition
                                pidx = index in self.rule_tree corresponding to the parent of rule
                                anode = (name of) vertex in rule's parent's RHS graph corresponding to rule
            cover = dict mapping ``vertex v`` -> ``index in decomposition of rule R that terminally covers v``
            transition_matrix = ignore for now
            temporal_matrix = ignore for now

        Class properties:
            root = the tuple (root_idx, root_rule)
            root_idx = the index (in the rule tree) of the root of the decomposition
            root_rule = the root rule of the decomposition
            mdl = the minimum description length (MDL) of the collection of rules
            dl = the minimum description length (MDL) of the collection of rules
            ll = the (conditional) log-likelihood of this grammar conditioned on the previous grammar
                 == 0 if this grammar is static
                 >= 0 if this grammar is dynamic
    """

    __slots__ = (
        'gtype', 'clustering', 'name', 'mu',
        'rules', 'decomposition', 'cover', 'extraction_map',
        # 'transition_matrix', 'temporal_matrix',
        'penalty', 'amplifier'
    )

    def __init__(self, gtype: str, clustering: str, name: str, mu: int):
        self.gtype: str = gtype
        self.clustering: str = clustering
        self.name: str = name
        self.mu: int = mu
        self.extraction_map: dict[int, int] = {}

        # self.rule_list
        # self.rule_dict
        self.rules: dict[int, list[BaseRule]] = {}
        self.decomposition: list[list] = []
        self.cover: dict[int, int] = {}
        # self.transition_matrix = None
        # self.temporal_matrix = None
        self.penalty: float = 0
        self.amplifier: float = 100

    @property
    def root(self) -> tuple[int, BaseRule]:
        for idx, (r, pidx, anode) in enumerate(self.decomposition):
            if pidx is None and anode is None:
                return idx, r
        raise AssertionError('decomposition does not have a root!')

    @property
    def root_idx(self) -> int:
        idx, _ = self.root
        return idx

    @property
    def root_rule(self) -> BaseRule:
        _, r = self.root
        return r

    @property
    def mdl(self) -> float:
        return self.dl

    @property
    def dl(self) -> float:
        return sum(sum(rule.mdl for rule in rules) for lhs, rules in self.rules.items())

    @property
    def ll(self) -> float:
        return np.log(self.likelihood)

    # the more modifications were required accommodate new rules, the lower the likelihood
    @property
    def likelihood(self) -> float:
        return 1 / (1 + self.cost + self.amplifier * self.penalty)  # adding 1 to the denominator avoids division by zero and ensures ∈ (0, 1]

    # total cost (in terms of edit operations) incurred to dynamically augment this grammar
    @property
    def cost(self) -> float:
        return sum(rule.edit_dist for rule, _, _ in self.decomposition)
        # return sum(sum(rule.edit_dist for rule in self.rule_list) for lhs, rules in self.rules)

    @property
    def nonterminals(self):
        if not self.rules:
            self.compute_rules()
        return set(self.rules)

    def compute_rules(self, merge: bool = False):
        for idx, (rule, _, _) in enumerate(self.decomposition):
            assert idx == rule.idn, f'{idx}, {rule.idn}, {rule}'
            rule.frequency = 1
        self.rules: dict[int, list[BaseRule]] = {}

        for rule, _, _ in self.decomposition:
            if rule.lhs in self.rules:
                if merge:  # merge isomorphic copies of the same rule together
                    for other_rule in self.rules[rule.lhs]:
                        if rule == other_rule:
                            rule.frequency = 0
                            other_rule.frequency += 1
                            break
                    else:
                        self.rules[rule.lhs] += [rule]
                else:  # distinguish between isomorphic copies of the same rule
                    self.rules[rule.lhs] += [rule]
            else:
                self.rules[rule.lhs] = [rule]

    def set_time(self, time: int):
        for _, (rule, _, _) in enumerate(self.decomposition):
            rule.time_created = time
            rule.time_changed = time

    def minimum_edit_dist(self, rule: BaseRule, parallel: bool = True, n_jobs: int = 4) -> int:
        if rule.lhs in self.nonterminals:
            candidates = [r for r in self.rule_dict[rule.lhs] if r is not rule]
            penalty = 0
        else:
            candidates = [r for r in self.rule_list if r is not rule]
            penalty = 1  # penalty for having to also modify the LHS symbol

        if len(candidates) == 0:
            return 0

        if parallel:
            edit_dists = Parallel(n_jobs=n_jobs)(
                delayed(graph_edit_distance)(r.graph, rule.graph) for r in candidates
            )
        else:
            edit_dists = [graph_edit_distance(r.graph, rule.graph) for r in candidates]

        return int(min(edit_dists)) + penalty

    def is_edge_connected(self, u: int, v: int) -> bool:
        return NotImplemented
        return u in self.cover or v in self.cover

    # find a reference to a rule somewhere in this grammar
    def find_rule(self, ref: Union[int, BaseRule]) -> int:
        if isinstance(ref, int):
            return ref
        refs = find(ref, self.decomposition)
        here, = refs if refs else [[]]
        return here

    # find the direct descendants downstream of this location in the decomposition
    def find_children(self, ref: Union[int, BaseRule]) -> list[tuple[int, BaseRule]]:
        idx = self.find_rule(ref)
        return [(cidx, r)
                for cidx, (r, pidx, _) in enumerate(self.decomposition)
                if idx == pidx]

    # find the direct descendants downstream of this nonterminal symbol in this location in the decomposition
    def find_children_of(self, nts: str, ref: Union[int, BaseRule]) -> list[tuple[int, BaseRule]]:
        idx = self.find_rule(ref)
        assert 'label' in self.decomposition[idx][0].graph.nodes[nts]  # type: ignore
        return [(cidx, r)
                for cidx, (r, pidx, anode) in enumerate(self.decomposition)
                if idx == pidx and nts == anode]

    def compute_levels(self):
        curr_level = 0
        root_idx, root_rule = self.root
        root_rule.level = curr_level
        children = self.find_children(root_idx)

        while len(children) != 0:
            curr_level += 1
            for _, crule in children:
                crule.level = curr_level
            children = [grandchild for cidx, crule in children for grandchild in self.find_children(cidx)]

    def level(self, ref: Union[int, BaseRule]) -> int:
        level = 0
        this_idx = self.find_rule(ref)

        parent_idx = self.decomposition[this_idx][1]

        while parent_idx is not None:
            level += 1
            parent_idx = self.decomposition[parent_idx][1]

        return level

    # replaces every occurrence of old_rule with new_rule in the grammar
    # f: V(old_rule) -> V(new_rule)
    def replace_rule(self, old_rule: BaseRule, new_rule: BaseRule, f: dict[str, str]):
        return NotImplemented
        for cidx, _ in self.find_children(old_rule):
            ancestor_node = self.decomposition[cidx][2]
            self.decomposition[cidx][2] = f[ancestor_node]
        replace(old_rule, new_rule, self.decomposition)

    def copy(self):
        vrg_copy = VRG(gtype=self.gtype, clustering=self.clustering, name=self.name, mu=self.mu)
        vrg_copy.decomposition = [[rule.copy(), pidx, anode] for rule, pidx, anode in self.decomposition]
        vrg_copy.cover = self.cover.copy()
        vrg_copy.extraction_map = self.extraction_map.copy()
        return vrg_copy

    def __len__(self):
        return len(self.decomposition)

    def __contains__(self, rule: BaseRule):
        return isinstance(self.find_rule(rule), int)

    def __str__(self):
        st = (
            f'graph: {self.name}, mu: {self.mu}, type: {self.gtype} clustering: {self.clustering} rules: {len(self.decomposition)}'
            f'({len(self)}) mdl: {round(self.mdl, 3):_g} bits'
        )
        return st

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        return self.decomposition[item]

    def reset(self):
        self.decomposition = []
        self.cover = {}
        self.extraction_map = {}
        # self.transition_matrix = None
        # self.temporal_matrix = None
        self.dl = 0

    # adds to the grammar iff it's a new rule
    def add_rule(self, rule: BaseRule) -> BaseRule:
        return NotImplemented
        if rule.lhs not in self.rule_dict:
            self.rule_dict[rule.lhs] = []

        for old_rule in self.rule_dict[rule.lhs]:
            nm = iso.categorical_node_match('label', '')  # does not take into account b_deg on nodes
            em = iso.numerical_edge_match('weight', 1.0)  # pylint: disable=not-callable
            gm = iso.GraphMatcher(old_rule.graph, rule.graph, node_match=nm, edge_match=em)

            # the isomorphism is given by gm.mapping if the graphs are isomorphic
            # f: V(old_rule.graph) -> V(rule.graph)
            # f⁻¹: V(rule.graph) -> V(old_rule.graph)
            if gm.is_isomorphic():
                f = gm.mapping
                f_inv = {fx: x for x, fx in f.items()}

                # merge the node attributes
                for old_v, old_d in old_rule.graph.nodes(data=True):
                    v = f[old_v]
                    if 'colors' in old_d.keys():
                        old_rule.graph.nodes[old_v]['colors'] += rule.graph.nodes[v]['colors']
                    if 'appears' in old_d.keys():
                        old_rule.graph.nodes[old_v]['appears'] += rule.graph.nodes[v]['appears']

                # merge the edge attributes
                for old_u, old_v, old_d in old_rule.graph.edges(data=True):
                    u = f[old_u]
                    v = f[old_v]
                    if 'colors' in old_d.keys():
                        old_rule.graph.edges[old_u, old_v]['colors'] += rule.graph.edges[u, v]['colors']
                    if 'attr_records' in old_d.keys():
                        old_rule.graph.edges[old_u, old_v]['attr_records'] += rule.graph.edges[u, v]['attr_records']

                # augment old_rule.mapping by extending it with f_inv ∘ rule.mapping
                for v in rule.mapping:
                    assert v not in old_rule.mapping  # the rules' vertex covers should be disjoint
                    old_rule.mapping[v] = f_inv[rule.mapping[v]]

                old_rule.frequency += 1
                old_rule.level = min(old_rule.level, rule.level)
                rule.idn = old_rule.idn  # why is this line here?
                self.replace_rule(rule, old_rule, f_inv)
                return old_rule

        # no pre-existing isomorphic rule was found
        rule.idn = len(self)
        self.rule_list.append(rule)
        self.rule_dict[rule.lhs].append(rule)
        return rule

    def init_temporal_matrix(self):
        return NotImplemented
        n = len(self.rule_list)
        self.temporal_matrix = np.identity(n, dtype=float)

        for idx, rule in enumerate(self.rule_list):
            self.temporal_matrix[idx, idx] *= rule.frequency

        return

    # def conditional_matrix_ll(self, axis: str = 'col'):
    #     assert axis in ['row', 'col']
    #     rule_matrix = self.temporal_matrix.copy()
    #     ll = 0

    #     for idx, _ in enumerate(self.rule_list):
    #         if axis == 'col':
    #             ax = rule_matrix[idx, :].copy()
    #         else:
    #             ax = rule_matrix[:, idx].copy()

    #         if len(ax[ax > 0]) > 0:
    #             ax = ax / ax.sum()
    #             ll += np.log(ax[ax > 0]).sum()
    #         else:
    #             pass

    #     return ll

    # ignore this for now
    # def compute_transition_matrix(self):
    #     n = len(self.rule_list)
    #     self.transition_matrix = np.zeros((n, n), dtype=float)

    #     for child_idx, child_rule in tqdm(enumerate(self.rule_list), total=n):
    #         for rule, source_idx, _ in self.rule_tree:
    #             if source_idx is not None and child_rule == rule:
    #                 parent_rule, _, _ = self.rule_tree[source_idx]
    #                 parent_idx = self.rule_list.index(parent_rule)
    #                 self.transition_matrix[parent_idx][child_idx] += 1

    #     return
