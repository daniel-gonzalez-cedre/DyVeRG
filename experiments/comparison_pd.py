from os import getcwd
from os.path import join
import sys
sys.path.extend(['.', '..'])

import git
import networkx as nx
from tqdm import trange
# from loguru import logger

# from evaluation.statistics import degree_distribution, spectrum, average_degree, triangle_count, clustering, transitivity
from evaluation.metrics import portrait_divergence
from src.data import load_data
from src.utils import mkdir


def compare_mmd(t: int, truegraph: nx.Graph, modeldataprefix: str, njobs: int = 1) -> list[float]:
    for modeltrial in trange(10, desc=f'time {t}'):
        try:
            modeldatafilename = f'{modeldataprefix}_{modeltrial}.edgelist'
            modelgraph = nx.read_edgelist(modeldatafilename)
        except FileNotFoundError:
            return []

        yield portrait_divergence(truegraph, modelgraph)


if __name__ == '__main__':
    model: str = input('what model? ').strip().lower()
    assert model in ('er', 'cl', 'sbm', 'graphrnn', 'verg', 'dyverg')

    if model == 'dyverg':
        mode: str = input('what learning mode? ').strip().lower()
        assert mode in ('dynamic', 'incremental')
    else:
        mode: str = input('what generating mode? ').strip().lower()
        assert mode in ('static', 'dynamic', 'incremental')

    dataset: str = input('what dataset? ').strip().lower()
    assert dataset in ('email-dnc', 'email-enron', 'email-eucore', 'facebook-links', 'coauth-dblp')

    numjobs: int = int(input('number of parallel jobs? ').strip().lower())

    mu = 4
    clustering = 'leiden'

    rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    graphdir = join(rootpath, f'results/graphs_{mode}/{model}/{dataset}/')
    resultdir = join(rootpath, 'results/mmds/')
    resultfilename = join(resultdir, f'{dataset}_{model}_{mode}.mmd')
    mkdir(resultdir)

    loaded = load_data(dataname=dataset)
    times = [t for t, _ in loaded]
    graphs = [g for _, g in loaded]

    with open(resultfilename, 'w') as outfile:
        outfile.write('time,trial,mmd\n')
        for time in range(0, len(times)):
            results = compare_mmd(time, graphs[time], join(graphdir, f'{time}'), numjobs)

            for trial, discrepancy in enumerate(results):
                outfile.write(f'{time},{trial},{discrepancy}\n')
                outfile.flush()
