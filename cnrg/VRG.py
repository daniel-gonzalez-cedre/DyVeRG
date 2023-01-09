"""
refactored VRG
"""
from typing import Union

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

from cnrg.Rule import MetaRule, Rule
from src.utils import find


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

            ruledict = <description here>
            decomposition = extraction tree for the grammar (a.k.a. decomposition for the input graph)
                            each entry looks like [rule, pidx, anode], where:
                                rule = the rule extracted at this point in the decomposition
                                pidx = index in self.rule_tree corresponding to the parent of rule
                                anode = (name of) vertex in rule's parent's RHS graph corresponding to rule
            cover = dict mapping ``timestep t`` -> (dict mapping ``vertex v`` -> ``index of rule R that terminally covers v``)
            times = list of timestamps [t0, t1, … tn] in the order they were incorporated into this grammar

            transition_matrix = ignore for now
            temporal_matrix = ignore for now

        Class properties:
            root = the tuple (root_idx, root_rule)
            root_idx = the index (in the rule tree) of the root of the decomposition
            root_rule = the root rule of the decomposition
            mdl = the minimum description length (MDL) of the collection of rules
            ll = the (conditional) log-likelihood of this grammar conditioned on the previous grammar
                 == 0 if this grammar is static
                 >= 0 if this grammar is dynamic
    """

    __slots__ = (
        'gtype', 'clustering', 'name', 'mu', 'extraction_map',
        'decomposition', 'cover', 'times', 'ruledict',
        # 'transition_matrix', 'temporal_matrix',
        'penalty', 'amplifier'
    )

    def __init__(self, gtype: str, clustering: str, name: str, mu: int):
        self.gtype: str = gtype
        self.clustering: str = clustering
        self.name: str = name
        self.mu: int = mu
        self.extraction_map: dict[int, int] = {}

        self.decomposition: list[list] = []
        self.cover: dict[int, dict[int, int]] = {}
        self.times: list[int] = []
        self.ruledict: dict[int, dict[int, list[MetaRule]]] = {}

        # self.penalty: float = 0
        # self.amplifier: float = 100

    @property
    def root(self) -> tuple[int, MetaRule]:
        for idx, (r, pidx, anode) in enumerate(self.decomposition):
            if pidx is None and anode is None:
                return idx, r
        raise AssertionError('decomposition does not have a root!')

    @property
    def root_idx(self) -> int:
        idx, _ = self.root
        return idx

    @property
    def root_rule(self) -> MetaRule:
        _, r = self.root
        return r

    @property
    def nonterminals(self):
        if not self.rules:
            self.compute_rules()
        return set(self.rules)

    @property
    def mdl(self) -> float:
        return sum(rule.mdl for rule, _, _ in self.decomposition)

    # @property
    # def dl(self) -> float:
    #     return sum(rule.mdl for rule, _, _ in self.decomposition)

    def ll(self, posterior: int, prior: int = None, verbose: bool = False) -> float:
        return np.log(self.likelihood(posterior, prior=prior, verbose=verbose))

    # the more modifications were required accommodate new rules, the lower the likelihood
    def likelihood(self, posterior: int, prior: int = None, verbose: bool = False) -> float:
        # return 1 / (1 + self.cost(time) + self.amplifier * self.penalty)  # adding 1 to the denominator avoids division by zero and ensures ∈ (0, 1]
        return 1 / (1 + self.cost(posterior, prior=prior, verbose=verbose))  # adding 1 to the denominator avoids division by zero and ensures ∈ (0, 1]

    # total cost (in terms of edit operations) incurred to dynamically augment this grammar
    def cost(self, posterior: int, prior: int = None, verbose: bool = False) -> float:
        if len(self.times) == 1:
            return np.inf
        if not prior:
            prior = self.times[self.times.index(posterior) - 1]
        S = sum(metarule.edits[prior, posterior]
                for metarule, _, _ in tqdm(self.decomposition, disable=(not verbose))
                if prior in metarule.times)  # TODO: parallelize this line?
        return S

    def ensure(self, t1, t2):
        self.ruledict[t2] = {}
        if t2 not in self.cover:
            self.cover[t2] = self.cover[t1].copy()
        for metarule, _, _ in self.decomposition:
            metarule.ensure(t1, t2)

    def compute_rules(self, time: int, merge: bool = True):
        candidates = [metarule[time] for metarule, _, _ in self.decomposition
                      if time in metarule.times]

        for rule in candidates:
            rule.frequency = 1

        self.ruledict[time] = {}

        for rule in candidates:
            if rule.lhs in self.ruledict[time]:
                if merge:  # merge isomorphic copies of the same rule together
                    for other_rule in self.ruledict[time][rule.lhs]:
                        if rule == other_rule:  # isomorphism up to differences in boundary degree
                            rule.frequency = 0
                            other_rule.frequency += 1
                            break
                    else:
                        self.ruledict[time][rule.lhs] += [rule]
                else:  # distinguish between isomorphic copies of the same rule
                    self.ruledict[time][rule.lhs] += [rule]
            else:
                self.ruledict[time][rule.lhs] = [rule]

    # find a reference to a rule somewhere in this grammar
    def find_rule(self, ref: Union[int, MetaRule]) -> int:
        if isinstance(ref, int):
            return ref
        # refs = [idx for idx, (metarule, _, _) in enumerate(self.decomposition) if metarule is ref]
        refs = find(ref, self.decomposition)
        here, = refs if refs else [[]]
        return here

    # find the direct descendants downstream of this location in the decomposition
    def find_children(self, ref: Union[int, MetaRule]) -> list[tuple[int, MetaRule]]:
        idx = self.find_rule(ref)
        return [(cidx, r)
                for cidx, (r, pidx, _) in enumerate(self.decomposition)
                if idx == pidx]

    # find the direct descendants downstream of this nonterminal symbol in this location in the decomposition
    def find_children_of(self, nts: str, ref: Union[int, MetaRule], time: int) -> list[tuple[int, MetaRule]]:
        idx = self.find_rule(ref)
        assert 'label' in self.decomposition[idx][0][time].graph.nodes[nts]  # type: ignore
        return [(cidx, r)
                for cidx, (r, pidx, anode) in enumerate(self.decomposition)
                if idx == pidx and nts == anode]

    def compute_levels(self):
        curr_level = 0

        root_idx, root_metarule = self.root
        root_metarule.level = curr_level
        children = self.find_children(root_idx)

        while len(children) != 0:
            curr_level += 1
            for _, cmetarule in children:
                cmetarule.level = curr_level
            children = [grandchild for cidx, _ in children for grandchild in self.find_children(cidx)]

    def level(self, ref: Union[int, MetaRule]) -> int:
        level = 0
        this_idx = self.find_rule(ref)

        parent_idx = self.decomposition[this_idx][1]

        while parent_idx is not None:
            level += 1
            parent_idx = self.decomposition[parent_idx][1]

        return level

    def copy(self) -> 'VRG':
        vrg_copy = VRG(gtype=self.gtype, clustering=self.clustering, name=self.name, mu=self.mu)
        vrg_copy.decomposition = [[rule.copy(), pidx, anode] for rule, pidx, anode in self.decomposition]
        vrg_copy.extraction_map = self.extraction_map.copy()
        vrg_copy.cover = self.cover.copy()
        vrg_copy.times = self.times.copy()
        return vrg_copy

    def __len__(self):
        return len(self.decomposition)

    def __contains__(self, rule: MetaRule):
        return isinstance(self.find_rule(rule), int)

    def __str__(self):
        st = (f'graph: {self.name}, ' +
              f'mu: {self.mu}, type: {self.gtype}, ' +
              f'clustering: {self.clustering}, ' +
              f'rules: {len(self.decomposition)}')
        return st

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        return self.decomposition[item]

    def reset(self):
        self.decomposition = []
        self.cover = {}
        self.extraction_map = {}
        self.dl = 0
        # self.transition_matrix = None
        # self.temporal_matrix = None

    # def init_temporal_matrix(self):
    #     return NotImplemented
    #     n = len(self.rule_list)
    #     self.temporal_matrix = np.identity(n, dtype=float)

    #     for idx, rule in enumerate(self.rule_list):
    #         self.temporal_matrix[idx, idx] *= rule.frequency

    #     return

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
