from os import getcwd, makedirs
from os.path import join
from loguru import logger

import git
import torch
import networkx as nx
torch.cuda.set_device(1)

from baselines.timers import fit_timer, gen_timer
from baselines.graphrnn.fit import fit
from baselines.graphrnn.gen import gen
from src.data import load_data
# from src.utils import mkdir


def write_graph(g, filepath, filename):
    nx.write_edgelist(g, join(filepath, filename))


dataset: str = input('dataset: ').lower()
mode: str = input('static or dynamic? ').lower()
num_gen: int = int(input('number of graphs to generate (at each timestep): ').lower())
try:
    start: int = int(input('start at index (default 0): ').lower())
except ValueError:
    start: int = 0
perturb: bool = False

assert dataset in ('email-dnc', 'email-enron', 'email-eucore', 'facebook-links')
assert mode in ('static', 'dynamic')
assert isinstance(num_gen, int)

rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
resultspath = f'results/graphs_{mode}/graphrnn/{dataset}'
# logpath = f'results/logs/graphrnn/{dataset}'
logpath = 'results/logs/'

makedirs(join(rootpath, resultspath), exist_ok=True)
makedirs(join(rootpath, logpath), exist_ok=True)
logger.add(join(rootpath, logpath, f'graphrnn_{dataset}_{mode}_timing.log'), mode='w')

loaded = load_data(dataset)
graphs = [g for _, g in loaded]

if mode == 'static':  # static generation
    for t, graph in enumerate(graphs):
        input_graphs = 10 * [graph]

        args, model, output = fit_timer(fit, logger)(input_graphs, nn='rnn')
        generated_graphs = gen_timer(gen, logger)(args=args, model=model, output=output, num_gen=num_gen)

        for trial, gen_graph in enumerate(generated_graphs):
            write_graph(gen_graph, join(rootpath, resultspath), f'{t}_{trial}.edgelist')
else:  # dynamic generation
    for t in range(len(graphs)):
        counter = 1
        input_graphs = graphs[:t + 1]

        while len(input_graphs) < 10:
            input_graphs.append(input_graphs[t - counter])
            counter += 1

        args, model, output = fit_timer(fit, logger)(input_graphs, nn='rnn')
        generated_graphs = gen_timer(gen, logger)(args=args, model=model, output=output, num_gen=num_gen)

        for trial, graph in enumerate(generated_graphs):
            write_graph(graph, join(rootpath, resultspath), f'{t}_{trial}.edgelist')
        # for trial, graph in enumerate(generated_graphs):
        #     write_graph(graph, join(rootpath, resultspath), f'{t}_{trial}.edgelist')
