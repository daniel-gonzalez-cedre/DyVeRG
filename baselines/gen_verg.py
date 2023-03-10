from os import getcwd, makedirs
from os.path import join
import sys
sys.path.extend(['.', '..'])

import git
import networkx as nx
from loguru import logger

from baselines.timers import fit_timer, gen_timer
from baselines.fit import VeRG
from dyverg.VRG import VRG
from src.data import load_data
from src.decomposition import decompose
# from src.adjoin_graph import update_grammar


def write_graph(g, filepath, filename):
    nx.write_edgelist(g, join(filepath, filename))


def gen(grammar: VRG, time: int, target_n, number: int = 1) -> list[nx.Graph]:
    generated = [grammar.generate(time, target_n, verbose=True) for _ in range(number)]
    return generated


dataset: str = input('dataset: ').lower()
mode = 'static'
num_gen: int = int(input('number of graphs to generate (at each timestep): ').lower())
try:
    start: int = int(input('start at index (default 0): ').lower())
except ValueError:
    start: int = 0

rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
resultspath = f'results/graphs_{mode}/verg/{dataset}'
logpath = 'results/logs/'

makedirs(join(rootpath, resultspath), exist_ok=True)
makedirs(join(rootpath, logpath), exist_ok=True)
logger.add(join(rootpath, logpath, f'verg_{dataset}_{mode}_timing.log'), mode='w')

loaded = load_data(dataset)
graphs = [g for _, g in loaded]

dyngrammar = fit_timer(decompose, logger)(graphs[0], time=0, name=dataset)
generated_graphs = gen_timer(gen, logger)(dyngrammar, time=0, target_n=graphs[0].order(), number=num_gen)
for trial, gen_graph in enumerate(generated_graphs):
    write_graph(gen_graph, join(rootpath, resultspath), f'0_{trial}.edgelist')

for t, graph in enumerate(graphs):
    grammar = fit_timer(decompose, logger)(graph, time=t, name=dataset)
    generated_graphs = gen_timer(gen, logger)(grammar, time=t, target_n=graph.order(), number=num_gen)

    for trial, gen_graph in enumerate(generated_graphs):
        write_graph(gen_graph, join(rootpath, resultspath), f'{t}_{trial}.edgelist')
