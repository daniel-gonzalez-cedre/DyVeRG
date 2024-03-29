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


def compare_pd(t: int, truegraph: nx.Graph, modeldataprefix: str, njobs: int = 1) -> list[float]:
    for modeltrial in trange(10, desc=f'{model} {dataset} {t}'):
        try:
            modeldatafilename = f'{modeldataprefix}_{modeltrial}.edgelist'
            modelgraph = nx.read_edgelist(modeldatafilename)
        except FileNotFoundError:
            return []

        if modelgraph.order() == 0:
            yield 1.0
        else:
            modelgraph = nx.convert_node_labels_to_integers(modelgraph)
            yield portrait_divergence(truegraph, modelgraph)


if __name__ == '__main__':
    model, mode, statname, dataset = (None, None, None, None)

    if len(sys.argv) > 1:
        model = sys.argv[1]
    if len(sys.argv) > 2:
        mode = sys.argv[2]
    if len(sys.argv) > 3:
        dataset = sys.argv[3]

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

    if not dataset:
        dataset: str = input('what dataset? ').strip().lower()

    assert dataset in ('email-dnc', 'email-enron', 'email-eucore', 'coauth-dblp', 'facebook-links')

    # numjobs: int = int(input('number of parallel jobs? ').strip().lower())

    mu = 4
    clustering = 'leiden'

    rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    graphdir = join(rootpath, f'results/graphs_{mode}/{model}/{dataset}/')
    resultdir = join(rootpath, 'results/pds/')
    resultfilename = join(resultdir, f'{dataset}_{model}_{mode}.pd')
    mkdir(resultdir)

    loaded = load_data(dataname=dataset)
    times = [t for t, _ in loaded]
    graphs = [g for _, g in loaded]

    with open(resultfilename, 'w') as outfile:
        outfile.write('time,trial,pd\n')
        for time in range(0, len(times)):
            if time > 12:
                break

            results = compare_pd(time, graphs[time], join(graphdir, f'{time}'))

            for trial, discrepancy in enumerate(results):
                outfile.write(f'{time},{trial},{discrepancy}\n')
                outfile.flush()
