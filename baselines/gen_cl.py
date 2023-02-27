from os import getcwd, makedirs
from os.path import join
import sys
sys.path.extend(['.', '..'])

import git
import networkx as nx
from loguru import logger

from baselines.timers import fit_timer, gen_timer
from baselines.fit import chung_lu
from src.data import load_data


def write_graph(g, filepath, filename):
    nx.write_edgelist(g, join(filepath, filename))


def gen(degrees: list[int], selfloops: bool, number: int = 1) -> list[nx.Graph]:
    generated = [nx.expected_degree_graph(degrees, selfloops=selfloops) for _ in range(number)]
    return generated


dataset: str = input('dataset: ').lower()
mode = 'static'
num_gen: int = int(input('number of graphs to generate (at each timestep): ').lower())
try:
    start: int = int(input('start at index (default 0): ').lower())
    logmode: str = 'w' if start <= 0 else 'a'
except ValueError:
    start: int = 0
    logmode: str = 'w'

assert dataset in ('email-dnc', 'email-enron', 'email-eucore', 'facebook-links', 'coauth-dblp')
assert isinstance(num_gen, int)

rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
resultspath = f'results/graphs_{mode}/cl/{dataset}'
logpath = 'results/logs/'

makedirs(join(rootpath, resultspath), exist_ok=True)
makedirs(join(rootpath, logpath), exist_ok=True)
logger.add(join(rootpath, logpath, f'cl_{dataset}_{mode}_timing.log'), mode=logmode)

loaded = load_data(dataset)
graphs = [g for _, g in loaded]

for t, graph in enumerate(graphs):
    w, sl = fit_timer(chung_lu, logger)(graph)
    generated_graphs = gen_timer(gen, logger)(w, sl, number=num_gen)

    for trial, gen_graph in enumerate(generated_graphs):
        write_graph(gen_graph, join(rootpath, resultspath), f'{t}_{trial}.edgelist')