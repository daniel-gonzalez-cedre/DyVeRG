from os import getcwd
from os.path import join
import sys
sys.path.extend(['.', '..'])

import git
import numpy as np
import networkx as nx
from tqdm import trange
# from loguru import logger

# from dyverg.LightMultiGraph import convert
from src.data import load_data
from src.decomposition import decompose
from src.adjoin_graph import update_grammar
from src.utils import mkdir


def evaluate_true_graph_incremental(t: int, basegraph: nx.Graph, truegraph: nx.Graph, switch: bool, njobs: int = 1) -> list[float]:
    for _ in trange(10, desc=f'time {t}'):
        basegrammar = decompose(basegraph, time=t - 1, name=dataset)
        truegrammar = update_grammar(basegrammar, basegraph, truegraph, t - 1, t, switch=switch)
        truescore = truegrammar.ll(prior=t - 1, posterior=t, njobs=njobs)
        yield truescore


# def evaluate_true_graph_dynamic(t: int, priorgraphs: nx.Graph, truegraph: nx.Graph, njobs: int = 1) -> list[float]:
#     basegrammar = decompose(priorgraphs[0], time=0, name=dataset)
#     for idx, nextgraph in enumerate(priorgraphs[1:]):
#         basegrammar = update_grammar(basegrammar, priorgraphs[idx], nextgraph, idx, idx + 1)
#     for _ in trange(10, desc=f'time {t}'):
#         truegrammar = update_grammar(basegrammar, priorgraphs[-1], truegraph, t - 1, t)
#         truescore = truegrammar.ll(prior=t - 1, posterior=t, njobs=njobs)
#         yield truescore


def evaluate_model_graph(t: int, basegraph: nx.Graph, modeldataprefix: str, switch: bool, njobs: int = 1) -> list[float]:
    for modeltrial in trange(10, desc=f'time {t}'):
        basegrammar = decompose(basegraph, time=t - 1, name=dataset)

        try:
            modeldatafilename = f'{modeldataprefix}_{modeltrial}.edgelist'
            modelgraph = nx.read_edgelist(modeldatafilename)
        except FileNotFoundError:
            return []

        if modelgraph.order() == 0:
            modelscore = 0
            for metarule in basegrammar:
                rhs = metarule[t - 1].graph
                modelscore += rhs.order()
                modelscore += sum(rhs.edges[u, v]['weight'] for u in rhs for v in rhs if (u, v) in rhs.edges()) / 2
            modelscore = np.log(1 / modelscore)
        else:
            modelgrammar = update_grammar(basegrammar, basegraph, modelgraph, t - 1, t, switch=switch)
            modelscore = modelgrammar.ll(prior=t - 1, posterior=t, njobs=njobs)
        yield modelscore


if __name__ == '__main__':
    model: str = input('what model? ').strip().lower()
    assert model in ('er', 'cl', 'sbm', 'graphrnn', 'verg', 'dyverg')

    if model == 'dyverg':
        # mode: str = input('what learning mode? ').strip().lower()
        mode = 'incremental'
        assert mode in ('dynamic', 'incremental')
    else:
        mode: str = input('what generating mode? ').strip().lower()
        assert mode in ('static', 'dynamic', 'incremental')

    dataset: str = input('what dataset? ').strip().lower()
    assert dataset in ('email-dnc', 'email-enron', 'email-eucore', 'facebook-links', 'coauth-dblp')

    set_switch: bool = input('flip the switch? ').strip().lower() in ('yes', 'y', 'switch')
    numjobs: int = int(input('number of parallel jobs? ').strip().lower())

    mu = 4
    clustering = 'leiden'

    rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    graphdir = join(rootpath, f'results/graphs_{mode}/{model}/{dataset}/')
    resultdir = join(rootpath, 'results/likelihoods/')
    resultfilename = join(resultdir, f'{dataset}_{model}_{mode}_{set_switch}.ll')
    mkdir(resultdir)

    loaded = load_data(dataname=dataset)
    times = [t for t, _ in loaded]
    graphs = [g for _, g in loaded]

    with open(resultfilename, 'w') as outfile:
        outfile.write('time,trial,likelihood\n')
        for time in range(1, len(times)):
            if time > 11:
                break
            if model == 'dyverg':
                results = evaluate_true_graph_incremental(time, graphs[time - 1], graphs[time], set_switch, njobs=numjobs)
                # if mode == 'incremental':
                #     results = evaluate_true_graph_incremental(time, graphs[time - 1], graphs[time], set_switch, njobs=numjobs)
                # else:
                #     results = evaluate_true_graph_dynamic(time, graphs[:time], graphs[time], njobs=numjobs)
            else:
                results = evaluate_model_graph(time, graphs[time - 1], join(graphdir, f'{time}'), set_switch, njobs=numjobs)

            for trial, score in enumerate(results):
                outfile.write(f'{time},{trial},{score}\n')
                outfile.flush()
