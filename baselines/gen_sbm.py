from os import getcwd, makedirs
from os.path import join
import sys
sys.path.extend(['.', '..'])

import git
import networkx as nx
import pyintergraph as pig
import graph_tool.all as gt
from loguru import logger

from baselines.timers import fit_timer, gen_timer
from baselines.fit import stochastic_blockmodel
from src.data import load_data


def write_graph(g, filepath, filename):
    nx.write_edgelist(g, join(filepath, filename))


def networkx_to_graphtool(graph_nx: nx.Graph):
    return pig.nx2gt(graph_nx)


def graphtool_to_networkx(graph_gt) -> nx.Graph:
    return pig.InterGraph.from_graph_tool(graph_gt).to_networkx()


# def fit(graph_nx: nx.Graph):
#     graph_gt = networkx_to_graphtool(graph_nx)
#     state = gt.minimize_blockmodel_dl(graph_gt)
#     return state


def gen(state, number: int = 1) -> list[nx.Graph]:
    generated = []
    for _ in range(number):
        gen_gt = gt.generate_sbm(state.b.a,
                                 gt.adjacency(state.get_bg(),
                                              state.get_ers()).T)
        gen_nx = graphtool_to_networkx(gen_gt)
        generated.append(gen_nx)
    return generated


dataset: str = input('dataset: ').lower()
# mode: str = input('static or dynamic? ').lower()
mode = 'static'
num_gen: int = int(input('number of graphs to generate (at each timestep): ').lower())
try:
    start: int = int(input('start at index (default 0): ').lower())
except ValueError:
    start: int = 0

assert dataset in ('email-dnc', 'email-enron', 'email-eucore', 'facebook-links', 'coauth-dblp')
assert isinstance(num_gen, int)

rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
resultspath = f'results/graphs_{mode}/sbm/{dataset}'
# logpath = f'results/logs/sbm/{dataset}'
logpath = 'results/logs/'

makedirs(join(rootpath, resultspath), exist_ok=True)
makedirs(join(rootpath, logpath), exist_ok=True)
logger.add(join(rootpath, logpath, f'sbm_{dataset}_{mode}_timing.log'), mode='w')

loaded = load_data(dataset)
graphs = [g for _, g in loaded]

for t, graph in enumerate(graphs):
    params = fit_timer(stochastic_blockmodel, logger)(graph)
    generated_graphs = gen_timer(gen, logger)(params, number=num_gen)

    for trial, gen_graph in enumerate(generated_graphs):
        write_graph(gen_graph, join(rootpath, resultspath), f'{t}_{trial}.edgelist')
