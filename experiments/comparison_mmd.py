from os import getcwd
from os.path import join
import sys
sys.path.extend(['.', '..'])

import git
import networkx as nx
from tqdm import trange
# from loguru import logger

from evaluation.statistics import degree_distribution, spectrum, average_degree, triangle_count, clustering, transitivity
from evaluation.metrics import maximum_mean_discrepancy
from src.data import load_data
from src.utils import mkdir


def compare_mmd(t: int, truegraph: nx.Graph, statistic, modeldataprefix: str, njobs: int = 1) -> list[float]:
    truestat = statistic(truegraph)
    modelstats = []

    for modeltrial in trange(10, desc=f'{model} {dataset} {statname} {t}'):
        try:
            modeldatafilename = f'{modeldataprefix}_{modeltrial}.edgelist'
            modelgraph = nx.read_edgelist(modeldatafilename)
        except FileNotFoundError:
            return []

        modelstats.append(statistic(modelgraph))

        yield maximum_mean_discrepancy([truestat], modelstats)


if __name__ == '__main__':
    model, mode, statname, dataset = (None, None, None, None)

    if len(sys.argv) > 1:
        model = sys.argv[1]
    if len(sys.argv) > 2:
        mode = sys.argv[2]
    if len(sys.argv) > 3:
        statname = sys.argv[3]
    if len(sys.argv) > 4:
        dataset = sys.argv[4]

    if not model:
        model: str = input('what model? ').strip().lower()

    assert model in ('er', 'cl', 'sbm', 'graphrnn', 'verg', 'dyverg')

    if not mode:
        if model == 'dyverg':
            mode: str = input('what learning mode? ').strip().lower()
            assert mode in ('dynamic', 'incremental')
        else:
            mode: str = input('what generating mode? ').strip().lower()
            assert mode in ('static', 'dynamic', 'incremental')

    assert mode in ('static', 'dynamic', 'incremental')

    if not statname:
        statname: str = input('what to measure? ').strip().lower().replace(' ', '-').replace('_', '-')

    assert statname in ('degree-distribution', 'spectrum',  # vectors
                        'average-degree', 'triangle-count', 'clustering', 'transitivity')  # scalars
    statfunc = (
        degree_distribution if statname == 'degree-distribution' else
        spectrum if statname == 'spectrum' else
        average_degree if statname == 'average-degree' else
        triangle_count if statname == 'triangle-count' else
        clustering if statname == 'clustering' else
        transitivity
    )

    if not dataset:
        dataset: str = input('what dataset? ').strip().lower()

    assert dataset in ('email-dnc', 'email-enron', 'email-eucore', 'facebook-links', 'coauth-dblp')

    # numjobs: int = int(input('number of parallel jobs? ').strip().lower())

    mu = 4
    clustering = 'leiden'

    rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    graphdir = join(rootpath, f'results/graphs_{mode}/{model}/{dataset}/')
    resultdir = join(rootpath, f'results/mmds/{statname}/')
    resultfilename = join(resultdir, f'{dataset}_{model}_{mode}_{statname}.mmd')
    mkdir(resultdir)

    loaded = load_data(dataname=dataset)
    times = [t for t, _ in loaded]
    graphs = [g for _, g in loaded]

    with open(resultfilename, 'w') as outfile:
        outfile.write('time,trial,mmd\n')
        for time in range(0, len(times)):
            results = compare_mmd(time, graphs[time], statfunc, join(graphdir, f'{time}'))

            for trial, discrepancy in enumerate(results):
                outfile.write(f'{time},{trial},{discrepancy}\n')
                outfile.flush()
