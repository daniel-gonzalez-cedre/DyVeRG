from os import getcwd, mkdir
from os.path import join

import git
import torch
import networkx as nx
torch.cuda.set_device(0)

from baselines.graphrnn.fit import fit
from baselines.graphrnn.gen import gen
from src.data import load_data
# from src.utils import mkdir


dataset: str = input('dataset: ').lower()
mode: str = input('static or dynamic? ').lower()
num_gen: int = input('number of graphs to generate (at each timestep): ').lower()
perturb: bool = False

assert dataset in ('email-dnc', 'email-enron', 'email-eucore', 'facebook-links')
assert mode in ('static', 'dynamic')
assert isinstance(num_gen, int)

rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
resultspath = f'results/graphs_{mode}/graphrnn/{dataset}'

# makedirs(path, exist_ok=True)

loaded = load_data(dataset)
graphs = [g for _, g in loaded]


def write_graph(g, filepath, filename):
    with open(join(filepath, filename), 'w') as outfile:
        nx.write_edgelist(g, outfile)


if mode == 'static':  # static generation
    for t, graph in enumerate(graphs):
        input_graphs = 10 * [graph]
        args, model, output = fit(input_graphs, nn='rnn')
        generated_graphs = gen(args=args, model=model, output=output, num_gen=num_gen)
        for trial, graph in enumerate(generated_graphs):
            write_graph(graph, resultspath, f'{t}_{trial}.edgelist')
else:  # dynamic generation
    for t in range(len(graphs)):
        counter = 1
        input_graphs = graphs[:t + 1]
        while len(input_graphs) < 10:
            input_graphs.append(input_graphs[t - counter])
            counter += 1
        args, model, output = fit(input_graphs, nn='rnn')
        generated_graphs = gen(args=args, model=model, output=output)
        for trial, graph in enumerate(generated_graphs):
            write_graph(graph, resultspath, f'{t}_{trial}.edgelist')
