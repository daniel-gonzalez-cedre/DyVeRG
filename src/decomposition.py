from typing import Iterable

import networkx as nx

from dyverg.Rule import MetaRule, Rule
from dyverg.VRG import VRG
from dyverg.Tree import create_tree
from dyverg.LightMultiGraph import convert
from dyverg.LightMultiGraph import LightMultiGraph as LMG
from dyverg.extract import MuExtractor
from dyverg.partitions import leiden, louvain


def create_splitting_rule(subgrammars: Iterable[VRG], time: int) -> MetaRule:
    rhs = nx.Graph()
    S = min(min(rule[time].lhs for rule, _, _ in subgrammar.decomposition)
            for subgrammar in subgrammars)

    for idx, subgrammar in enumerate(subgrammars):
        rhs.add_node(str(idx), b_deg=0, label=subgrammar.root_rule[time].lhs)

    # S = sum(d['b_deg'] for v, d in rhs.nodes(data=True) if 'label' in d)
    return MetaRule(rules={time: Rule(S - 1, rhs)})


def decompose(g: nx.Graph, time: int = -1, mu: int = 4, clustering: str = 'leiden', gtype: str = 'mu_level_dl', name: str = '', verbose: bool = False):
    def merge_subgrammars(splitting_rule: MetaRule, subgrammars: list[VRG]) -> VRG:
        supergrammar = subgrammars[0]

        # make the root of the old decomposition point to the new root
        splitting_rule.idn = len(supergrammar.decomposition)
        prev_root_idx = supergrammar.root_idx
        supergrammar.decomposition[prev_root_idx][1] = splitting_rule.idn
        supergrammar.decomposition[prev_root_idx][2] = '0'
        # for idx, (_, pidx, anode) in enumerate(supergrammar.decomposition):
        #     if pidx is None and anode is None:
        #         supergrammar.decomposition[idx][1] = splitting_rule.idn
        #         supergrammar.decomposition[idx][2] = '0'
        #         break
        # else:
        #     raise AssertionError('never found the root rule')
        supergrammar.decomposition += [[splitting_rule, None, None]]

        for i, subgrammar in enumerate(subgrammars[1:], start=1):
            offset = len(supergrammar.decomposition)

            # shift the indices of the sub-decomposition
            for idx, (_, pidx, anode) in enumerate(subgrammar.decomposition):
                subgrammar.decomposition[idx][0].idn += offset
                if pidx is None and anode is None:
                    subgrammar.decomposition[idx][1] = splitting_rule.idn
                    subgrammar.decomposition[idx][2] = str(i)
                else:
                    subgrammar.decomposition[idx][1] += offset

            # append the sub-decomposition to the super-decomposition
            supergrammar.decomposition += subgrammar.decomposition

            # shift the indices of the covering index map
            assert len(supergrammar.cover[time].keys() & subgrammar.cover[time].keys()) == 0
            for node in subgrammar.cover[time]:
                subgrammar.cover[time][node] += offset
            supergrammar.cover[time] |= subgrammar.cover[time]

        return supergrammar

    if g.order() == 0:
        raise AssertionError('!!! graph is empty !!!')

    if float(nx.__version__[:3]) < 2.4:
        connected_components = nx.connected_component_subgraphs(g)
    else:
        connected_components = [g.subgraph(comp) for comp in nx.connected_components(g)]

    if len(connected_components) == 1:
        supergrammar = decompose_component(g, mu=mu, time=time, clustering=clustering,
                                           gtype=gtype, name=name, verbose=verbose)
    else:
        subgrammars = [decompose_component(component, mu=mu, time=time, clustering=clustering,
                                           gtype=gtype, name=name, verbose=verbose)
                       for component in connected_components]

        splitting_rule = create_splitting_rule(subgrammars, time)
        supergrammar = merge_subgrammars(splitting_rule, subgrammars)

    # sanity check
    for v in g.nodes():
        assert v in supergrammar.cover[time]

    # supergrammar.compute_rules()
    # supergrammar.compute_rules(time)
    supergrammar.compute_levels()
    supergrammar.times += [time]

    return supergrammar


def decompose_component(g: nx.Graph, mu: int = 4, time: int = -1, clustering: str = 'leiden', gtype: str = 'mu_level_dl', name: str = '', verbose: bool = False):
    if not isinstance(g, LMG):
        g = convert(g)

    if clustering == 'leiden':
        clusters = leiden(g)
    elif clustering == 'louvain':
        clusters = louvain(g)
    else:
        raise NotImplementedError

    dendrogram = create_tree(clusters)

    vrg = VRG(clustering=clustering,
              gtype=gtype,
              name=name,
              mu=mu)

    extractor = MuExtractor(g=g.copy(),
                            gtype=gtype,
                            root=dendrogram,
                            grammar=vrg,
                            mu=mu,
                            time=time)

    extractor.generate_grammar(verbose=verbose)
    grammar = extractor.grammar

    return grammar
