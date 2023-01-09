"""
VRG extraction
"""
import abc
import math
# import pickle
import itertools
# import logging
from typing import List, Tuple, Dict, Set, Any, Union

from tqdm import tqdm
import networkx as nx

# import sys
# sys.path.append('..')
# import cnrg

from cnrg.LightMultiGraph import LightMultiGraph
from cnrg.MDL import graph_dl
from cnrg.Rule import MetaRule, Rule
from cnrg.globals import find_boundary_edges
from cnrg.part_info import set_boundary_degrees
from cnrg.Tree import TreeNode
from cnrg.VRG import VRG


class Record:
    __slots__ = ('tnodes_list', 'rule_idn', 'frequency', 'boundary_edges_list', 'subtree_list', 'score')

    def __init__(self, rule_idn: int):
        self.tnodes_list: List[TreeNode] = []
        self.rule_idn: int = rule_idn
        self.frequency: int = 0  # number of times we have seen this rule_idn
        self.boundary_edges_list: List[Set[Tuple[int, int]]] = []
        self.subtree_list: List[Set[int]] = []
        self.score = None  # score of the rule

    def update(self, boundary_edges: Any, subtree: Set[int], tnode: TreeNode):
        self.frequency += 1
        self.boundary_edges_list.append(tuple(boundary_edges))
        self.subtree_list.append(tuple(subtree))
        self.tnodes_list.append(tnode)

    def remove(self):
        self.frequency -= 1

    def __repr__(self):
        st = ''
        if self.frequency == 0:
            st += '[x] '
        st += f'{self.rule_idn} > {self.tnodes_list}'
        return st

    def __str__(self):
        st = ''
        if self.frequency == 0:
            st += '[x] '
        st += f'{self.rule_idn} > {self.tnodes_list} {round(self.score, 3)}'
        return st


def create_rule(subtree: Set[int], g: LightMultiGraph) -> tuple[Rule, list[tuple[int, int]], dict[int, str]]:
    sg = g.subgraph(subtree).copy()
    assert isinstance(sg, LightMultiGraph)
    boundary_edges = find_boundary_edges(g, subtree)

    rule = Rule(lhs=len(boundary_edges), graph=sg, subtree=subtree)
    set_boundary_degrees(g, rule.graph)
    rule.generalize_rhs()

    return rule, boundary_edges


def compress_graph(g: LightMultiGraph, subtree: Set[int], boundary_edges: Any, permanent: bool) -> Union[None, float]:
    """
    :param g: the graph
    :param subtree: the set of nodes that's compressed
    :param boundary_edges: boundary edges
    :param permanent: if disabled, undo the compression after computing the new dl -> returns the float
    :return:
    """
    assert len(subtree) > 0, f'Empty subtree g:{g.order(), g.size()}, bound: {boundary_edges}'
    before = (g.order(), g.size())

    if not isinstance(subtree, set):
        subtree = set(subtree)

    if boundary_edges is None:
        boundary_edges = find_boundary_edges(g, subtree)

    # step 1: remove the nodes from subtree, keep track of the removed edges
    removed_nodes: set[int] = set()
    removed_edges: set[tuple[int, int]] = set()
    if not permanent:
        removed_nodes = list(g.subgraph(subtree).nodes(data=True))  # type: ignore
        removed_edges = list(g.subgraph(subtree).edges(data=True))  # type: ignore
    g.remove_nodes_from(subtree)
    new_node = min(subtree)

    # step 2: replace subtree with new_node
    g.add_node(new_node, label=len(boundary_edges))

    # step 3: rewire new_node
    for bdry in boundary_edges:
        if len(bdry) == 2:
            u, v = bdry
            if u in subtree:
                u = new_node
            if v in subtree:
                v = new_node
            g.add_edge(u, v)
        elif len(bdry) == 3:
            u, v, d = bdry
            if u in subtree:
                u = new_node
            if v in subtree:
                v = new_node
            g.add_edge(u, v, d)

    if not permanent:  # if this flag is set, then return the dl of the compressed graph and undo the changes
        compressed_graph_dl = graph_dl(g)
        g.remove_node(new_node)  # and the boundary edges
        g.add_nodes_from(removed_nodes)  # add the subtree

        for e in itertools.chain(removed_edges, boundary_edges):
            if len(e) == 3:
                u, v, d = e
            else:
                u, v = e
                d = {'weight': 1}
            if 'attr_records' in d.keys():
                g.add_edge(u, v, weight=d['weight'], attr_records=d['attr_records'])
            elif 'colors' in d.keys():
                g.add_edge(u, v, weight=d['weight'], colors=d['colors'])
            else:
                g.add_edge(u, v, weight=d['weight'])

        after = (g.order(), g.size())
        assert before == after, 'Decompression did not work'
        return compressed_graph_dl

    return None


class BaseExtractor(abc.ABC):
    # __slots__ = 'gtype', 'g', 'root', 'tnode_to_score', 'grammar', 'mu'

    def __init__(self, g: LightMultiGraph, gtype: str, root: TreeNode, grammar: VRG, mu: int, time: int = -1) -> None:
        assert gtype in (
            'local_dl',
            'global_dl',
            'mu_random',
            'mu_level',
            'mu_dl',
            'mu_level_dl'
        ), f'Invalid mode: {gtype}'
        self.g: nx.Graph = g  # the graph
        self.gtype: str = gtype
        self.root: TreeNode = root
        self.tnode_to_score: Dict[TreeNode, Any] = {}
        self.grammar: VRG = grammar
        self.mu: int = mu
        self.time: int = time
        self.extracted_sequence: list[MetaRule] = []  # stores list of rules in the order they were extracted

    def __str__(self) -> str:
        return f'Type: {self.gtype}, mu: {self.mu}'

    def __repr__(self) -> str:
        return str(self)

    def get_sorted_tnodes(self):
        tnodes, _ = zip(*sorted(self.tnode_to_score.items(), key=lambda kv: kv[1]))
        return tnodes

    def get_best_tnode_and_score(self) -> Any:
        """
        returns the tnode with the lowest score
        :return: tnode
        """
        return min(self.tnode_to_score.items(), key=lambda kv: kv[1])  # use the value as the key

    def update_subtree_scores(self, start_tnode: TreeNode) -> Any:
        """
        updates scores of the tnodes of the subtree rooted at start_tnode depending on the extraction type
        :param start_tnode: starting tnode. for the entire tree, use self.root
        :return:
        """
        active_nodes = set(self.g.nodes())
        stack: List[TreeNode] = [start_tnode]
        nodes_visited = 0

        while len(stack) != 0:
            tnode = stack.pop()
            nodes_visited += 1
            subtree = tnode.leaves & active_nodes

            score = self.tnode_score(tnode=tnode, subtree=subtree)
            self.tnode_to_score[tnode] = score

            for kid in tnode.kids:
                if not kid.is_leaf:  # don't add to the bucket if it's a leaf
                    stack.append(kid)
                # perc = (nodes_visited / total_tree_nodes) * 100
                # progress = perc - pbar.n
                # pbar.update(progress)
        return

    def update_ancestor_scores(self, tnode: TreeNode):
        """
        updates the scores of the ancestors
        :param tnode:
        :return:
        """
        active_nodes = set(self.g.nodes())
        tnode_leaves = tnode.leaves
        new_tnode_key = min(tnode_leaves)
        old_tnode_key = tnode.key
        tnode_children = tnode.children
        is_global_extractor = hasattr(self, 'rule_idn_to_record')

        tnode = tnode.parent
        while tnode is not None:
            subtree = tnode.leaves & active_nodes
            tnode.leaves -= tnode_leaves
            tnode.leaves.add(new_tnode_key)  # tnode becomes a new leaf

            tnode.children.discard(old_tnode_key)  # remove the old tnode key from all subsequent ancestors  # switched from remove to discard
            tnode.children -= tnode_children
            tnode.children.add(new_tnode_key)
            if not is_global_extractor:
                self.tnode_to_score[tnode] = self.tnode_score(tnode=tnode, subtree=subtree)
            tnode = tnode.parent
        return

    @abc.abstractmethod
    def update_tree(self, tnode: TreeNode) -> None:
        """
        update the tree as needed - ancestors and descendants
        :param tnode:
        :return:
        """
        return NotImplemented

    @abc.abstractmethod
    def tnode_score(self, tnode: TreeNode, subtree: Set[int]) -> Any:
        """
        computes the score of a subtree
        :param subtree:
        :return:
        """
        return NotImplemented

    @abc.abstractmethod
    def extract_rule(self) -> MetaRule:
        """
        extracts one rule using the Extraction method
        :return:
        """
        return NotImplemented

    def generate_grammar(self, verbose: bool = False) -> None:
        """
        generates the grammar
        """
        num_nodes = self.g.order()
        self.grammar.cover[self.time] = {}

        with tqdm(total=100, bar_format='{l_bar}{bar}|[{elapsed}<{remaining}]', ncols=50, disable=(not verbose)) as pbar:
            while True:
                metarule = self.extract_rule()
                assert nx.is_connected(self.g), 'graph is disconnected'
                assert metarule is not None

                # rule = self.grammar.add_rule(rule)
                self.extracted_sequence += [metarule]

                percent = (1 - (self.g.order() - 1) / (num_nodes - 1)) * 100 if num_nodes > 1 else 100
                curr_progress = percent - pbar.n
                pbar.update(curr_progress)

                if metarule[self.time].lhs == 0:  # we are compressing the root, so that's the end
                    assert self.g.order() == 1, 'Graph not correctly compressed'
                    break


class MuExtractor(BaseExtractor):
    def __init__(self, g: LightMultiGraph, gtype: str, root: TreeNode, grammar: VRG, mu: int, time: int = 0):
        super().__init__(g=g, gtype=gtype, root=root, grammar=grammar, mu=mu, time=time)
        self.grammar: VRG
        self.update_subtree_scores(start_tnode=self.root)  # initializes the scores

    def tnode_score(self, tnode: TreeNode, subtree: Set[int]) -> Union[float, Tuple[float, int], Tuple[float, int, float]]:
        """
        returns infinity for rules > mu
        :param tnode:
        :param subtree:
        :return:
        """
        score = None
        diff = tnode.get_num_leaves() - self.mu

        if diff > 0:  # there are more nodes than mu: rank oversized rules smallest to largest
            # mu_score = float('inf')
            mu_score = 1000000.0 + tnode.get_num_leaves()
        elif diff < 0:
            mu_score = math.log2(1 - diff)  # mu is greater
        else:
            mu_score = 0  # no penalty

        if self.gtype == 'mu_random':
            score = mu_score  # |mu - nleaf|
        elif self.gtype == 'mu_level':
            score = mu_score, tnode.level  # |mu - nleaf|, level of the tnode
        elif 'dl' in self.gtype:  # compute cost only if description length is used for scores
            if diff > 0:  # don't bother creating the rule
                rule_cost = None
            else:
                temp_rule, _ = create_rule(subtree=subtree, g=self.g)
                rule_cost = temp_rule.mdl

            if self.gtype == 'mu_dl':
                score = mu_score, rule_cost
            elif self.gtype == 'mu_level_dl':
                score = mu_score, tnode.level, rule_cost

        assert score is not None, 'score is None'
        return score

    def update_tree(self, tnode: TreeNode) -> None:
        """
        In this case, only update ancestors and their scores
        :param tnode:
        :return:
        """
        new_key = min(tnode.leaves)
        self.update_ancestor_scores(tnode=tnode)

        # delete score entries for all the subtrees
        del self.tnode_to_score[tnode]  # tnode now is a leaf
        for child in filter(lambda x: isinstance(x, str), tnode.children):
            del self.tnode_to_score[child]
        tnode.make_leaf(new_key=new_key)

    def extract_rule(self) -> MetaRule:
        """
        Step 1: get best tnode
        Step 2: create rule, add to grammar
        Step 3: compress graph, update tree
        :return:
        """

        best_tnode, _ = self.get_best_tnode_and_score()
        subtree = best_tnode.leaves & set(self.g.nodes())

        rule, boundary_edges = create_rule(subtree=subtree, g=self.g)
        rule.idn = len(self.grammar.decomposition)

        # !!! CRITICAL SECTOR !!!
        metarule = MetaRule(rules={self.time: rule}, idn=rule.idn)
        self.grammar.decomposition.append([metarule, None, None])

        for x, d in self.g.subgraph(subtree).nodes(data=True):
            if 'label' in d:  # if x is a nonterminal symbol
                child_idx = self.grammar.extraction_map[x]  # find the rule corresponding to x
                self.grammar.decomposition[child_idx][1] = metarule.idn  # make this rule the parent of that rule
                self.grammar.decomposition[child_idx][2] = metarule[self.time].alias[x]  # map that child rule to this RHS node
                # self.grammar.push_down_branch(self.grammar.decomposition[child_idx][0])  # increase descendants' levels
            else:  # if x is a regular node
                self.grammar.cover[self.time][x] = metarule.idn  # associate that node with this rule in the decomposition

        self.grammar.extraction_map[min(subtree)] = metarule.idn  # map compressed node to this index in the decomposition
        # !!! CRITICAL SECTOR !!!

        compress_graph(g=self.g, subtree=subtree, boundary_edges=boundary_edges, permanent=True)
        self.update_tree(tnode=best_tnode)

        return metarule
