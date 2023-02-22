from os import getcwd
from os.path import join
import sys
from argparse import ArgumentParser
sys.path.append('..')

import git
import networkx as nx
from tqdm import tqdm

from experiments.generate_static import verg_generate, uniform_generate, er_generate, cl_generate
from evaluation.statistics import average_degree, triangle_count, clustering, transitivity
from evaluation.statistics import degree_distribution, spectrum
from evaluation.metrics import maximum_mean_discrepancy as mmd
from evaluation.metrics import portrait_divergence as pd
from src.data import load_data
from src.utils import mkdir, transpose


modelname = {
    verg_generate: 'verg',
    uniform_generate: 'uniform',
    er_generate: 'erdos-renyi',
    cl_generate: 'chung-lu'
}
modelfunc = {val: key for (key, val) in modelname.items()}

statname = {
    average_degree: 'degreeavg',
    triangle_count: 'triangle',
    clustering: 'clustering',
    transitivity: 'transitivity',
    degree_distribution: 'degreedist',
    spectrum: 'spectrum'
}
statfunc = {val: key for (key, val) in statname.items()}

all_models = [verg_generate, uniform_generate, er_generate, cl_generate]
all_scalars = [average_degree, triangle_count, clustering, transitivity]
all_vectors = [degree_distribution, spectrum]


def main():
    rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    resultspath = f'results/graphs_static/{args.model}/{args.dataset}'
    mkdir(join(rootpath, resultspath))

    loaded = load_data(dataname=args.dataset)
    # times = [t for t, _ in loaded]
    graphs_data = [g for _, g in loaded]
    model = modelfunc[args.model]

    # use transpose
    graphs_gen = [model(graphs_data, verbose=args.verbose) for _ in range(args.trials)]
    graphs_gen = transpose(graphs_gen)

    for idx, graphs in enumerate(graphs_gen):
        for t, graph_t in enumerate(graphs):
            with open(join(rootpath, resultspath, f'{idx}_{t}.edgelist'), 'wb') as graphfile:
                nx.write_edgelist(graph_t, graphfile)

    return
    ######

    # for t in range(1, args.trials + 1):
    #     graphs_gen = model(graphs_data, verbose=args.verbose)
    #     for idx, graph in enumerate(graphs_gen):
    #         with open(join(rootpath, resultspath, f'{idx}_{t}.edgelist'), 'wb') as graphfile:
    #             nx.write_edgelist(graph, graphfile)

    #     if args.portrait or args.statistic == '':  # use portrait divergence
    #         with open(join(rootpath, resultspath, f'static_pd/{args.dataset}_{modelname[model]}_{t}.pd'), 'w') as outfile:
    #             for idx, (g_data, g_gen) in enumerate(zip(graphs_data, graphs_gen)):
    #                 outfile.write(f'{idx},{pd(g_data, g_gen)}\n')
    #     else:  # use maximum mean discrepancy
    #         statistic = statfunc[args.statistic]
    #         stats_gen = [([statistic(graph)] if statistic in all_scalars else statistic(graph)) for graph in graphs_gen]
    #         stats_data = [([statistic(graph)] if statistic in all_scalars else statistic(graph)) for graph in graphs_data]

    #         with open(join(rootpath, resultspath, f'static_mmd/{args.dataset}_{modelname[model]}_{statname[statistic]}_{t}.mmd'), 'w') as outfile:
    #             for idx, (data, gen) in enumerate(zip(stats_data, stats_gen)):
    #                 outfile.write(f'{idx},{mmd([data], [gen])}\n')

    #         with open(join(rootpath, resultspath, f'static_mmd/{args.dataset}_{modelname[model]}_{statname[statistic]}.mmd_whole'), 'w') as outfile:
    #             outfile.write(f'{mmd(stats_data, stats_gen)}\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model',
                        type=str,
                        choices=['verg', 'uniform', 'erdos-renyi', 'chung-lu'],
                        help="select a model from ['verg', 'uniform', 'erdos-renyi', 'chung-lu']")
    parser.add_argument('dataset',
                        type=str,
                        choices=['email-dnc', 'email-enron', 'email-eucore', 'facebook-links'],
                        help="select a dataset from ['email-dnc', 'email-enron', 'email-eucore', 'facebook-links']")
    parser.add_argument('-t', '--trials',
                        default=1,
                        dest='trials',
                        type=int,
                        help="number of times to generate each graph")
    # parser.add_argument('-s', '--statistic',
    #                     type=str,
    #                     default='',
    #                     dest='statistic',
    #                     choices=['degreeavg', 'triangle', 'clustering', 'transitivity', 'degreedist', 'spectrum'],
    #                     help="select a statistic from ['degreeavg', 'triangle', 'clustering', 'transitivity', 'degreedist', 'spectrum']")
    parser.add_argument('-p', '--portrait',
                        action='store_true',
                        default=False,
                        dest='portrait',
                        help='whether or not to use portrait divergence as the dissimilarity metric')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        dest='verbose',
                        help='verbosity')
    args = parser.parse_args()
    main()
