import time
from os import getcwd, makedirs
from os.path import join
import sys
sys.path.extend(['.', '..'])
sys.setrecursionlimit(10**6)

import git
import networkx as nx
from loguru import logger
from joblib import Parallel, delayed

from baselines.timers import fit_timer, gen_timer
from baselines.fit import DyVeRG
from dyverg.VRG import VRG
from src.data import load_data
from src.decomposition import decompose
from src.adjoin_graph import update_grammar


def write_graph(g, filepath, filename):
    nx.write_edgelist(g, join(filepath, filename))


def gen(grammar: VRG, time: int, target_n: int, number: int = 1) -> list[nx.Graph]:
    generated = [grammar.generate(time, target_n, verbose=True) for _ in range(number)]
    return generated


def gen_parallel(grammar: VRG, time: int, target_n: int) -> nx.Graph:
    return grammar.generate(time, target_n, verbose=True)


def work(num, func, *args, **kwargs) -> tuple[int, float, nx.Graph]:
    start = time.process_time()
    results = func(*args, **kwargs)
    end = time.process_time()
    return num, end - start, results


dataset: str = input('dataset: ').lower()
num_gen: int = int(input('number of graphs to generate (at each timestep): ').lower())
try:
    start: int = int(input('start at index (default 0, incremental -1): ').lower())
except ValueError:
    start: int = 0
parallel: bool = input('parallel? ') in ('yes', 'y', 'parallel', 'p', '')

mode = 'dynamic' if start >= 0 else 'incremental'

rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
resultspath = f'results/graphs_{mode}/dyverg/{dataset}'
logpath = 'results/logs/'

makedirs(join(rootpath, resultspath), exist_ok=True)
makedirs(join(rootpath, logpath), exist_ok=True)
logger.add(join(rootpath, logpath, f'dyverg_{dataset}_{mode}_timing.log'), mode='w')

loaded = load_data(dataset)
graphs = [g for _, g in loaded]

if mode == 'dynamic':
    dyngrammar = fit_timer(decompose, logger)(graphs[0], time=0, name=dataset)
    generated_graphs = gen_timer(gen, logger)(dyngrammar, time=0, target_n=graphs[0].order(), number=num_gen)
    for trial, gen_graph in enumerate(generated_graphs):
        write_graph(gen_graph, join(rootpath, resultspath), f'0_{trial}.edgelist')

    for t in range(1, len(graphs)):
        dyngrammar = fit_timer(update_grammar, logger)(dyngrammar, graphs[t - 1], graphs[t], t - 1, t, switch=False)

        if parallel:
            with Parallel(n_jobs=num_gen) as parallel:
                for trial, gen_time, gen_graph in parallel(
                    delayed(work)(num, gen_parallel, dyngrammar, time=t, target_n=graphs[t].order())
                    for num in range(num_gen)
                ):
                    logger.info('gen time elapsed: {gen_time}', gen_time=gen_time)
                    write_graph(gen_graph, join(rootpath, resultspath), f'{t}_{trial}.edgelist')
        else:
            generated_graphs = gen_timer(gen, logger)(dyngrammar, time=t, target_n=graphs[t].order(), number=num_gen)

            for trial, gen_graph in enumerate(generated_graphs):
                write_graph(gen_graph, join(rootpath, resultspath), f'{t}_{trial}.edgelist')
else:
    for t in range(0, len(graphs) - 1):
        basegrammar = fit_timer(decompose, logger)(graphs[t], time=t, name=dataset)
        dyngrammar = fit_timer(update_grammar, logger)(basegrammar, graphs[t], graphs[t + 1], t, t + 1, switch=False)

        if parallel:
            with Parallel(n_jobs=num_gen) as parallel:
                for trial, gen_time, gen_graph in parallel(
                    delayed(work)(num, gen_parallel, dyngrammar, time=t + 1, target_n=graphs[t + 1].order())
                    for num in range(num_gen)
                ):
                    logger.info('gen time elapsed: {gen_time}', gen_time=gen_time)
                    write_graph(gen_graph, join(rootpath, resultspath), f'{t + 1}_{trial}.edgelist')
        else:
            generated_graphs = gen_timer(gen, logger)(dyngrammar, time=t + 1, target_n=graphs[t + 1].order(), number=num_gen)

            for trial, gen_graph in enumerate(generated_graphs):
                write_graph(gen_graph, join(rootpath, resultspath), f'{t + 1}_{trial}.edgelist')
