from os import getcwd, makedirs
from os.path import join
import sys
sys.path.extend(['.', '..'])

import git
import networkx as nx
from loguru import logger

from baselines.timers import fit_timer, gen_timer
from baselines.fit import erdos_renyi
from src.data import load_data


def write_graph(g, filepath, filename):
    nx.write_edgelist(g, join(filepath, filename))


def gen(n: int, p: float, directed: bool = False, number: int = 1) -> list[nx.Graph]:
    generated = [nx.erdos_renyi_graph(n, p, directed=directed) for _ in range(number)]
    return generated


dataset: str = input('dataset: ').lower()
mode = 'static'
num_gen: int = int(input('number of graphs to generate (at each timestep): ').lower())
try:
    start: int = int(input('start at index (default 0): ').lower())
except ValueError:
    start: int = 0

assert dataset in ('email-dnc', 'email-enron', 'email-eucore', 'facebook-links', 'coauth-dblp')
assert isinstance(num_gen, int)

rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
resultspath = f'results/graphs_{mode}/er/{dataset}'
logpath = 'results/logs/'

makedirs(join(rootpath, resultspath), exist_ok=True)
makedirs(join(rootpath, logpath), exist_ok=True)
logger.add(join(rootpath, logpath, f'er_{dataset}_{mode}_timing.log'), mode='w')

loaded = load_data(dataset)
graphs = [g for _, g in loaded]

for t, graph in enumerate(graphs):
    params = fit_timer(erdos_renyi, logger)(graph, directed=False)
    generated_graphs = gen_timer(gen, logger)(params[0], params[1], directed=params[2], number=num_gen)

    for trial, gen_graph in enumerate(generated_graphs):
        write_graph(gen_graph, join(rootpath, resultspath), f'{t}_{trial}.edgelist')
