from os import getcwd, makedirs
from os.path import join
import sys
sys.path.extend(['.', '..'])

import git
import torch
import networkx as nx
from loguru import logger
torch.cuda.set_device(0)

from baselines.fit import graphRNN
from baselines.graphrnn.train import test_rnn_epoch
from baselines.timers import fit_timer, gen_timer
from src.data import load_data
# from src.utils import mkdir


def write_graph(g, filepath, filename):
    nx.write_edgelist(g, join(filepath, filename))


def gen(args, model, output, number=1):
    for generated in test_rnn_epoch(0, args, model, output, test_batch_size=number):
        yield generated


dataset: str = input('dataset: ').strip().lower()
mode: str = input('static, dynamic, or incremental? ').strip().lower()
num_gen: int = int(input('number of graphs to generate (at each timestep): ').strip().lower())
try:
    start: int = int(input('start at index (default: 0): ').strip().lower())
    logmode: str = 'w' if start <= 0 else 'a'
except ValueError:
    start: int = 0
    logmode: str = 'w'
perturb: bool = False

assert dataset in ('email-dnc', 'email-enron', 'email-eucore', 'facebook-links', 'coauth-dblp')
assert mode in ('static', 'dynamic', 'incremental')
assert isinstance(num_gen, int)

rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
resultspath = f'results/graphs_{mode}/graphrnn/{dataset}'
# logpath = f'results/logs/graphrnn/{dataset}'
logpath = 'results/logs/'

makedirs(join(rootpath, resultspath), exist_ok=True)
makedirs(join(rootpath, logpath), exist_ok=True)
logger.add(join(rootpath, logpath, f'graphrnn_{dataset}_{mode}_timing.log'), mode=logmode)

loaded = load_data(dataset)
graphs = [g for _, g in loaded]

if mode == 'static':  # static generation
    for t, graph in enumerate(graphs):
        input_graphs = 10 * [graph]

        params = fit_timer(graphRNN, logger)(input_graphs, nn='rnn')
        generated_graphs = gen_timer(gen, logger)(args=params[0], model=params[1], output=params[2], number=num_gen)

        for trial, gen_graph in enumerate(generated_graphs):
            write_graph(gen_graph, join(rootpath, resultspath), f'{t}_{trial}.edgelist')
elif mode == 'dynamic':  # dynamic generation
    for t in range(len(graphs)):
        counter = 1
        input_graphs = graphs[:t + 1]

        while len(input_graphs) < 10:
            input_graphs.append(input_graphs[t - counter])
            counter += 1

        params = fit_timer(graphRNN, logger)(input_graphs, nn='rnn')
        generated_graphs = gen_timer(gen, logger)(args=params[0], model=params[1], output=params[2], number=num_gen)

        for trial, graph in enumerate(generated_graphs):
            write_graph(graph, join(rootpath, resultspath), f'{t}_{trial}.edgelist')
        # for trial, graph in enumerate(generated_graphs):
        #     write_graph(graph, join(rootpath, resultspath), f'{t}_{trial}.edgelist')
else:  # incremental generation
    for t, (graph1, graph2) in enumerate(zip(graphs[:-1], graphs[1:])):
        input_graphs = (5 * [graph1]) + (5 * [graph2])

        params = fit_timer(graphRNN, logger)(input_graphs, nn='rnn')
        generated_graphs = gen_timer(gen, logger)(args=params[0], model=params[1], output=params[2], number=num_gen)

        for trial, gen_graph in enumerate(generated_graphs):
            write_graph(gen_graph, join(rootpath, resultspath), f'{t}_{trial}.edgelist')
