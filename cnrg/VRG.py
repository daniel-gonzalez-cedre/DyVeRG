"""
refactored VRG
"""

# import os
# import copy
# from collections import defaultdict
from typing import List, Dict

import numpy as np
import networkx.algorithms.isomorphism as iso
from tqdm import tqdm

# import src.full_info as full_info
# import src.no_info as no_info
# import src.part_info as part_info
from cnrg.Rule import PartRule


class VRG:
    """
    Class for Vertex Replacement Grammars
    """

    __slots__ = (
        'name',
        'type',
        'clustering',
        'mu',
        'rule_list',
        'rule_dict',
        'cost',
        'num_rules',
        'rule_tree',
        'rule_source',
        'which_rule_source',
        'comp_map',
        'transition_matrix',
        'temporal_matrix'
    )

    def __init__(self, type, clustering, name, mu):
        self.name: str = name  # name of the graph
        self.type: str = type  # type of grammar - lambda, local, global, selection strategy - random, dl, level, or dl_levels
        self.clustering: str = clustering  # clustering strategy
        self.mu = mu

        self.rule_list: List[PartRule] = []  # list of Rule objects
        self.rule_dict: Dict[int, List[PartRule]] = {}  # dictionary of rules, keyed in by their LHS
        self.cost: int = 0  # the MDL of the rules
        self.num_rules: int = 0  # number of active rules
        self.rule_tree = []  # extraction tree for the grammar; each entry looks like [rule, parent_idx, which_child]
        self.rule_source = {}  # dict that maps vertex v -> index in rule_tree of rule R that terminally covers v
        self.which_rule_source = {}  # dict that maps vertex v -> index in list(R.graph.nodes()), where R = self.rule_source[v], indicating which node in the rule RHS corresponds to v
        self.comp_map = {}
        self.transition_matrix = None
        self.temporal_matrix = None

    # ignore this for now
    def compute_transition_matrix(self):
        n = len(self.rule_list)
        self.transition_matrix = np.zeros((n, n), dtype=float)

        for child_idx, child_rule in tqdm(enumerate(self.rule_list), total=n):
            for rule, source_idx, _ in self.rule_tree:
                if source_idx is not None and child_rule == rule:
                    parent_rule, _, _ = self.rule_tree[source_idx]
                    parent_idx = self.rule_list.index(parent_rule)
                    self.transition_matrix[parent_idx][child_idx] += 1

        return

    def init_temporal_matrix(self):
        n = len(self.rule_list)
        self.temporal_matrix = np.identity(n, dtype=float)

        for idx, rule in enumerate(self.rule_list):
            self.temporal_matrix[idx, idx] *= rule.frequency

        return

    # ignore this for now
    def transition_ll(self):
        raise NotImplementedError
        return

    def conditional_ll(self, axis: str = 'col'):
        assert axis in ['row', 'col']
        rule_matrix = self.temporal_matrix.copy()
        ll = 0

        for idx, rule in enumerate(self.rule_list):
            if axis == 'col':
                ax = rule_matrix[idx, :].copy()
            else:
                ax = rule_matrix[:, idx].copy()

            if len(ax[ax > 0]) > 0:
                ax = ax / ax.sum()
                ll += np.log(ax[ax > 0]).sum()
            else:
                pass

        return ll

    def copy(self):
        vrg_copy = VRG(type=self.type, clustering=self.clustering, name=self.name, mu=self.mu)
        vrg_copy.cost = self.cost
        vrg_copy.num_rules = self.num_rules

        # new attributes also need to be copied over
        vrg_copy.rule_tree = [[rule.copy(), parent_idx, which_idx] for rule, parent_idx, which_idx in self.rule_tree]

        vrg_copy.rule_list = list({rule for rule, _, _ in self.rule_tree})
        vrg_copy.rule_dict = {lhs: [rule for rule in vrg_copy.rule_list if rule.lhs == lhs] for lhs in [rule.lhs for rule in self.rule_list]}

        vrg_copy.rule_source = {key: value for key, value in self.rule_source.items()}
        vrg_copy.which_rule_source = {key: value for key, value in self.which_rule_source.items()}
        vrg_copy.comp_map = {key: value for key, value in self.comp_map.items()}
        vrg_copy.transition_matrix = self.transition_matrix.copy() if self.transition_matrix is not None else None
        vrg_copy.temporal_matrix = self.temporal_matrix.copy() if self.temporal_matrix is not None else None

        return vrg_copy

    def __len__(self):
        return len(self.rule_list)

    def __contains__(self, rule: PartRule):
        return rule in self.rule_dict[rule.lhs]

    def __str__(self):
        if self.cost == 0:
            self.calculate_cost()
        st = (
            f'graph: {self.name}, mu: {self.mu}, type: {self.type} clustering: {self.clustering} rules: {len(self.rule_list):_d}'
            f'({self.num_rules:_d}) mdl: {round(self.cost, 3):_g} bits'
        )
        return st
        # return f'{self.name}, mode: {self.mode} clustering: {self.clustering} selection: {} lambda: {} rules: {}({}) mdl: {} bits'.format(self.name, self.mode, self.clustering, self.selection,
        #                                                                                                 self.lamb, self.active_rules, len(self.rule_list), round(self.cost, 3))

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        return self.rule_list[item]

    def reset(self):
        # reset the grammar
        self.rule_list = []
        self.rule_dict = {}
        self.cost = 0
        self.num_rules = 0

    def add_rule(self, rule: PartRule) -> int:
        # adds to the grammar iff it's a new rule
        if rule.lhs not in self.rule_dict:
            self.rule_dict[rule.lhs] = []

        for old_rule in self.rule_dict[rule.lhs]:
            if rule == old_rule:  # check for isomorphism
                g1 = old_rule.graph
                g2 = rule.graph

                gm = iso.GraphMatcher(old_rule.graph, rule.graph)
                assert gm.is_isomorphic()

                for old_v, dd in old_rule.graph.nodes(data=True):
                    v = gm.mapping[old_v]
                    if 'node_colors' in dd.keys():
                        old_rule.graph.nodes[old_v]['node_colors'] += rule.graph.nodes[v]['node_colors']
                    if 'appears' in dd.keys():
                        old_rule.graph.nodes[old_v]['appears'] += rule.graph.nodes[v]['appears']

                # for old_v, v in gm.mapping.items():
                #     if 'node_colors' in old_rule.graph.nodes[old_v].keys() and 'node_colors' in rule.graph.nodes[v].keys():
                #         old_rule.graph.nodes[old_v]['node_colors'] += rule.graph.nodes[v]['node_colors']

                for old_u, old_v, dd in old_rule.graph.edges(data=True):
                    u = gm.mapping[old_u]
                    v = gm.mapping[old_v]
                    if 'edge_colors' in dd.keys():
                        old_rule.graph.edges[old_u, old_v]['edge_colors'] += rule.graph.edges[u, v]['edge_colors']
                    if 'attr_records' in dd.keys():
                        old_rule.graph.edges[old_u, old_v][
                            'attr_records'
                        ] += rule.graph.edges[u, v]["attr_records"]

                old_rule.frequency += 1
                rule.id = old_rule.id
                return old_rule.id

        # if I'm going to allow for deletions, there needs to be a better way to number things to prevent things from getting clobbered
        rule.id = self.num_rules
        # new rule
        self.num_rules += 1

        self.rule_list.append(rule)
        self.rule_dict[rule.lhs].append(rule)
        return rule.id

    # def deactivate_rule(self, rule_id):
    #     """
    #     deletes the rule with rule_id from the grammar
    #     :param rule_id:
    #     :return:
    #     """
    #     # do not decrease num_rules
    #     rule = self.rule_list[rule_id]
    #     rule.deactivate()
    #     # TODO check if rule deactivation propagates to the dictionary
    #     # self.rule_dict[rule.lhs]

    def calculate_cost(self):
        self.cost = 0
        for rule in self.rule_list:
            rule.calculate_cost()
            self.cost += rule.cost
        return self.cost
