"""
refactored VRG
"""
from typing import Union

from joblib import Parallel, delayed
import numpy as np
import networkx.algorithms.isomorphism as iso
from tqdm import tqdm

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

            rule_list = list of Rule objects
            rule_dict = dictionary of rules, keyed in by their LHS
            dl = the minimum description length (MDL) of the collection of rules
            rule_tree = extraction tree for the grammar (a.k.a. decomposition for the input graph)
                        each entry looks like [rule, parent_idx, ancestor_node], where:
                            rule = the rule extracted at this point in the decomposition
                            parent_idx = index in self.rule_tree corresponding to the parent of rule
                            ancestor_node = (name of) vertex in rule's parent's RHS graph corresponding to rule
            rule_source = dict mapping ``vertex v`` -> ``index in rule_tree of rule R that terminally covers v``
            which_rule_source = dict that maps vertex v -> index in list(R.graph.nodes()), where:
                                    R = self.rule_source[v],
                                    indicating which node in the rule RHS corresponds to v
            comp_map = maps the dendrogram nodes to the rules extracted from them
                       used during the extraction process to compute rule_tree
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
        'gtype', 'clustering', 'name', 'mu', 'rule_list', 'rule_dict',
        'rule_tree', 'covering_idx', 'comp_map',
        'transition_matrix', 'temporal_matrix'
    )

    def __init__(self, gtype: str, clustering: str, name: str, mu: int):
        self.gtype: str = gtype  # type of grammar - lambda, local, global, selection strategy - random, dl, level, or dl_levels
        self.clustering: str = clustering  # clustering strategy
        self.name: str = name  # name of the graph
        self.mu = mu

        self.rule_list: list[BaseRule] = []  # list of Rule objects
        self.rule_dict: dict[int, list[BaseRule]] = {}  # dictionary of rules, keyed in by their LHS
        self.rule_tree: list[list] = []  # extraction tree for the grammar; each entry looks like [rule, parent_idx, which_child]
        self.covering_idx: dict[int, int] = {}  # dict that maps vertex v -> index in rule_tree of rule R that terminally covers v
        self.comp_map: dict[int, int] = {}
        self.transition_matrix = None
        self.temporal_matrix = None

    @property
    def root(self) -> tuple[int, BaseRule]:
        for idx, (r, pidx, anode) in enumerate(self.rule_tree):
            if pidx is None and anode is None:
                assert r.lhs == min(nts for nts in self.rule_dict)
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
        return sum(rule.mdl for rule in self.rule_list)

    @property
    def ll(self) -> float:
        return np.log(1 / (1 + sum(rule.edit_dist for rule in self.rule_list)))

    def minimum_edit_dist(self, rule: BaseRule, parallel: bool = True, n_jobs: int = 4) -> int:
        if rule.lhs in self.rule_dict:
            candidates = [r for r in self.rule_dict[rule.lhs] if r is not rule]
            penalty = 0
        else:
            candidates = [r for r in self.rule_list if r is not rule]
            penalty = 1  # penalty for having to also modify the LHS symbol

        if parallel:
            edit_dists = Parallel(n_jobs=n_jobs)(
                delayed(graph_edit_distance)(r.graph, rule.graph) for r in candidates
            )
        else:
            edit_dists = [graph_edit_distance(r.graph, rule.graph) for r in candidates]

        return int(min(edit_dists)) + penalty

    # find a reference to a rule somewhere in this grammar
    def find_rule(self, rule: BaseRule, where: str) -> Union[int, tuple[int, int], list[int]]:
        assert where in ['rule_list', 'list',
                         'rule_dict', 'dict',
                         'rule_tree', 'tree', 'decomposition']
        where = where.strip().split('_')[-1]

        if where == 'list':
            refs = find(rule, self.rule_list)
            here, = refs if refs else [[]]

        elif where == 'dict':
            refs = find(rule, self.rule_dict)
            (lhs, idxs), = refs if refs else (((), ()),)
            here = (lhs, *idxs)

        elif where in ('tree', 'decomposition'):
            here = [idx for idx, _ in find(rule, self.rule_tree)]

        else:
            here = []

        return here

    # find all direct descendants of (all copies of) this rule in the decomposition
    def get_all_children(self, rule: BaseRule) -> list[tuple[int, BaseRule]]:
        refs: list[int] = self.find_rule(rule, where='rule_tree')  # type: ignore
        return [(cidx, r)
                for ref in refs
                for cidx, (r, pidx, _) in enumerate(self.rule_tree)
                if ref == pidx]

    # find all direct descendants corresponding to the nonterminal symbol in (all copies of) this rule
    def get_all_children_of(self, nts: int, rule: BaseRule) -> list[tuple[int, BaseRule]]:
        assert 'label' in rule.graph.nodes[nts]
        refs: list[int] = self.find_rule(rule, where='rule_tree')  # type: ignore
        return [(cidx, r)
                for ref in refs
                for cidx, (r, pidx, anode) in enumerate(self.rule_tree)
                if ref == pidx and nts == anode]

    # find the direct descendants downstream of this location in the decomposition
    def get_children(self, ref: int) -> list[tuple[int, BaseRule]]:
        return [(cidx, r)
                for cidx, (r, pidx, _) in enumerate(self.rule_tree)
                if ref == pidx]

    # find the direct descendants downstream of this nonterminal symbol in this location in the decomposition
    def get_children_of(self, nts: int, ref: int) -> list[tuple[int, BaseRule]]:
        assert 'label' in self.rule_tree[ref][0].graph.nodes[nts]  # type: ignore
        return [(cidx, r)
                for cidx, (r, pidx, anode) in enumerate(self.rule_tree)
                if ref == pidx and nts == anode]

    # TODO: during extraction, when rules are merged, take the min of their levels
    # find the minimum distance in the decomposition between this rule and the root
    def find_level(self, r: BaseRule) -> int:
        return min(self.find_levels(r))

    # find all distances in the decomposition between this rule and the root
    def find_levels(self, r: BaseRule) -> list[int]:
        levels = []
        for this_idx in self.find_rule(r, where='rule_tree'):  # type: ignore
            level = 0
            parent_idx = self.rule_tree[this_idx][1]

            while parent_idx is not None:
                level += 1
                parent_idx = self.rule_tree[parent_idx][1]

            levels += [level]
        return levels

    # replaces every occurrence of old_rule with new_rule in the grammar
    # f: V(old_rule) -> V(new_rule)
    def replace_rule(self, old_rule: BaseRule, new_rule: BaseRule, f: dict[str, str]):
        for child_idx, _ in self.get_all_children(old_rule):
            ancestor_node = self.rule_tree[child_idx][2]
            self.rule_tree[child_idx][2] = f[ancestor_node]
        replace(old_rule, new_rule, self.rule_tree)
        replace(old_rule, new_rule, self.rule_dict)
        replace(old_rule, new_rule, self.rule_list)

    def push_down_grammar(self, steps: int = 1):
        self.push_down_branch(self.root_rule, steps=steps)
        # for rule in self.rule_list:
        #     rule.level += steps

    def push_down_branch(self, rule: BaseRule, steps: int = 1):
        rule.level += steps
        for _, child in self.get_all_children(rule):
            self.push_down_branch(child)

    def copy(self):
        bag = [rule.copy() for rule in self.rule_list]

        vrg_copy = VRG(gtype=self.gtype, clustering=self.clustering, name=self.name, mu=self.mu)

        vrg_copy.rule_tree = [[bag[bag.index(rule)], parent_idx, ancestor_node] for rule, parent_idx, ancestor_node in self.rule_tree]

        vrg_copy.rule_list = [bag[idx] for idx, _ in enumerate(self.rule_list)]
        vrg_copy.rule_dict = {lhs: [rule for rule in bag if rule.lhs == lhs] for lhs in [rule.lhs for rule in bag]}
        # vrg_copy.rule_dict = {lhs: [bag[bag.index(rule)] for rule in self.rule_dict[lhs]] for lhs in self.rule_dict}

        vrg_copy.covering_idx = self.covering_idx.copy()
        vrg_copy.comp_map = self.comp_map.copy()
        vrg_copy.transition_matrix = self.transition_matrix.copy() if self.transition_matrix is not None else None
        vrg_copy.temporal_matrix = self.temporal_matrix.copy() if self.temporal_matrix is not None else None

        return vrg_copy

    def __len__(self):
        return len(self.rule_list)

    def __contains__(self, rule: BaseRule):
        return rule in self.rule_dict[rule.lhs]

    def __str__(self):
        st = (
            f'graph: {self.name}, mu: {self.mu}, type: {self.gtype} clustering: {self.clustering} rules: {len(self.rule_list)}'
            f'({len(self)}) mdl: {round(self.mdl, 3):_g} bits'
        )
        return st

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        return self.rule_list[item]

    def reset(self):
        self.rule_list = []
        self.rule_dict = {}
        self.rule_tree = []
        self.covering_idx = {}
        self.comp_map = {}
        self.transition_matrix = None
        self.temporal_matrix = None
        self.dl = 0

    # adds to the grammar iff it's a new rule
    def add_rule(self, rule: BaseRule) -> BaseRule:
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
