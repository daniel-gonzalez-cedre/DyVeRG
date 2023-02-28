"""
refactored VRG
"""
from typing import Union

# from joblib import Parallel, delayed
from tqdm import tqdm
# import networkx as nx
import numpy as np

from dyverg.LightMultiGraph import LightMultiGraph
from dyverg.Rule import MetaRule, Rule
from src.utils import find, boundary_edges


class VRG:
    """
        Vertex-replacement graph grammar

        Class attributes:
            clustering = node clustering algorithm used on the input graph to extract this grammar
            mu = the μ hyperparameter that (indirectly) determines the rules' maximum RHS sizes
            name = a string naming the graph this grammar was extracted from
            gtype = a string describing the extraction strategy for the grammar

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
        'mu', 'clustering', 'name', 'gtype',
        'extraction_map',
        'decomposition', 'cover', 'times', 'ruledict',
        # 'transition_matrix', 'temporal_matrix',
        # 'penalty', 'amplifier'
    )

    def __init__(self, mu: int, clustering: str = 'leiden', name: str = ''):
        self.clustering: str = clustering
        self.mu: int = mu
        self.name: str = name
        self.gtype: str = 'mu_level_dl'

        self.extraction_map: dict[int, int] = {}

        self.decomposition: list[list] = []
        self.cover: dict[int, dict[int, int]] = {}
        self.times: list[int] = []
        # self.ruledict: dict[int, dict[int, list[MetaRule]]] = {}

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
    def mdl(self) -> float:
        return sum(rule.mdl for rule, _, _ in self.decomposition)

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
                for metarule, _, _ in tqdm(self.decomposition, desc='computing edits', disable=(not verbose))
                if prior in metarule.times)  # TODO: parallelize this line?
        return S

    def ensure(self, time):
        for t in self.times:
            if t < time:
                for metarule, _, _ in self.decomposition:
                    if t not in metarule.times:
                        metarule[t] = Rule(lhs=0, graph=LightMultiGraph(), idn=metarule.idn)

        if time in self.times:
            return

        if time not in self.cover:
            self.cover[time] = self.cover[max(self.times)].copy()

        for metarule, _, _ in self.decomposition:
            assert time not in metarule.times
            metarule[time] = metarule[max(metarule.times)].copy()

        # self.ruledict[time] = {}
        self.times.append(time)

    def compute_rules(self, time: int, merge: bool = True) -> dict[int, list[tuple[Rule, int]]]:
        # self.ruledict[time] = {}
        ruledict = {}  # lhs |-> [(rule, freq), ...]
        candidates = [metarule[time] for metarule, _, _ in self.decomposition
                      if time in metarule.times]  # if (time in metarule.times) and (metarule[time].graph.order() != 0)]

        # for rule in candidates:
        #     rule.frequency = 1

        for rule in candidates:
            if rule.lhs in ruledict:
                if merge:  # merge isomorphic copies of the same rule together
                    for idx, (other_rule, freq) in enumerate(ruledict[rule.lhs]):
                        if rule == other_rule:  # isomorphism up to differences in boundary degree
                            # rule.frequency = 0
                            # other_rule.frequency += 1
                            ruledict[rule.lhs][idx][1] = freq + 1
                            break
                    else:
                        ruledict[rule.lhs].append([rule, 1])
                else:  # distinguish between isomorphic copies of the same rule
                    ruledict[rule.lhs].append([rule, 1])
            else:
                ruledict[rule.lhs] = [[rule, 1]]

        for ll in ruledict.values():
            for idx, tt in enumerate(ll):
                ll[idx] = tuple(tt)

        return ruledict

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

    def generate(self, time: int, goal: int, multigraph: bool = False,
                 tolerance: float = 0.05, merge_rules: bool = True,
                 rule_order: bool = False, verbose: bool = False) -> tuple[LightMultiGraph, list[int]]:
        try:
            lower_bound = int(goal * (1 - tolerance))
            upper_bound = int(goal * (1 + tolerance))
            max_attempts = 100000

            ruledict = self.compute_rules(time, merge=merge_rules)
            for _ in tqdm(range(max_attempts), desc='timeout meter', disable=(not verbose)):
                g, ro = self._generate(ruledict, upper_bound)

                if (g is not None) and (lower_bound <= g.order() <= upper_bound):
                    # if verbose:
                    #     tqdm.write(f'Generation succeeded after {attempt} attempts.')
                    if rule_order:
                        return g, ro
                    return g

            raise TimeoutError(f'Generation failed after exceeding {max_attempts} attempts.')
        except TimeoutError:
            return self.generate(time, goal,
                                 tolerance=(2 * tolerance), merge_rules=merge_rules,
                                 rule_order=rule_order, verbose=verbose)

    def _generate(self, ruledict, upper_bound) -> tuple[LightMultiGraph, list[int]]:
        node_counter = 1
        rng = np.random.default_rng()

        S = min(ruledict)  # find the starting symbol
        nonterminals = [0]  # names of nodes in g corresponding to nonterminal symbols
        rule_ordering = []  # idn's of rules in the order they were applied

        g = LightMultiGraph()
        g.add_node(0, label=S)

        while len(nonterminals) > 0:
            if g.order() > upper_bound:
                return None, None

            # choose a nonterminal symbol at random
            nts: int = rng.choice(nonterminals)
            lhs: int = g.nodes[nts]['label']
            candidate_rules: list[Rule] = [rr for rr, _ in ruledict[lhs]]
            candidate_freqs: list[int] = [ff for _, ff in ruledict[lhs]]

            # select a new rule to apply
            freqs = np.asarray(candidate_freqs)
            weights = freqs / np.sum(freqs)
            rule = rng.choice(candidate_rules, p=weights).copy()  # we will have to modify the boundary degrees
            rhs = rule.graph

            rule_ordering.append(rule.idn)
            broken_edges: list[tuple[int, int]] = boundary_edges(g, {nts})
            try:
                assert len(broken_edges) == max(0, lhs)
            except:
                import pdb
                pdb.set_trace()

            g.remove_node(nts)
            nonterminals.remove(nts)

            # add all of the nodes from the rule to the graph
            node_map = {}
            for n, d in rhs.nodes(data=True):
                new_node = node_counter
                node_map[n] = new_node
                attr = {'b_deg': d['b_deg']}

                if 'label' in d:
                    attr['label'] = d['label']
                    nonterminals.append(new_node)

                if 'colors' in d:
                    attr['color'] = rng.choice(d['colors'])

                g.add_node(new_node, **attr)
                node_counter += 1

            # add all of the edges from the rule to the graph
            for u, v, d in rhs.edges(data=True):
                attr = {'weight': d['weight']}
                if 'colors' in d:
                    attr['color'] = rng.choice(d['colors'])

                g.add_edge(node_map[u], node_map[v], **attr)

            # rewire the broken edges from g to the new structure from the rule
            while len(broken_edges) > 0:
                eidx = rng.choice(len(broken_edges))
                edge = broken_edges.pop(eidx)
                u, v, *d = edge

                # choose a node on the rule's right-hand side to attach this broken edge to
                n = rng.choice([x for x, d in rhs.nodes(data=True) if d['b_deg'] > 0])
                rhs.nodes[n]['b_deg'] -= 1

                # there should never be self-edges on nonterminal symbols
                if u == nts and v != nts:
                    u = node_map[n]
                elif u != nts and v == nts:
                    v = node_map[n]
                else:
                    raise AssertionError(f'investigate: {nts}, {u}, {v}, {edge}')

                # attach the nonterminal we previously selected to the rule node
                g.add_edge(u, v, **dict(*d))

        return g, rule_ordering

    def copy(self) -> 'VRG':
        vrg_copy = VRG(mu=self.mu, clustering=self.clustering, name=self.name)
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
              f'mu: {self.mu}, ' +
              f'clustering: {self.clustering}, ' +
              f'rules: {len(self)}')
        return st

    def __repr__(self):
        return str(self)

    def __iter__(self):
        for metarule, _, _ in self.decomposition:
            yield metarule

    def __getitem__(self, item):
        return self.decomposition[item][0]

    def reset(self):
        self.decomposition = []
        self.cover = {}
        self.extraction_map = {}
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
