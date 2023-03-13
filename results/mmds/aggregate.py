import sys
from os import getcwd
from os.path import join
sys.path.extend(['./', '../', '../../'])

from tqdm import tqdm
import git
import numpy as np
import scipy.stats as st


def confidence(data: list[float], level: float = 0.95):
    return st.sem(data) * st.t.ppf((1 + level) / 2.0, len(data) - 1)


def interval(data: list[float]):
    return st.t.interval(0.95, len(data) - 1, loc=np.mean(data), scale=st.sem(data))


# def mean_ci(data: list[float], level: float = 0.95):
#     data = 1.0 * np.array(data)
#     mean, se = np.mean(data), st.sem(data)
#     h = se * st.t.ppf((1 + level) / 2.0, len(data) - 1)
#     return mean, mean - h, mean + h

if __name__ == '__main__':
    rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    models = ['er', 'cl', 'sbm', 'graphrnn', 'verg', 'dyverg']
    datasets = ['email-dnc', 'email-enron', 'email-eucore', 'facebook-links', 'coauth-dblp']
    statistic = input('statistic: ').strip().lower().replace('_', '-').replace(' ', '-')
    assert statistic in ('degree-distribution', 'spectrum', 'average-degree', 'clustering', 'transitivity', 'triangle-count')

    # dname = input('dataset: ').strip().lower()
    # assert dname in datasets

    for dataset in tqdm(datasets):
    # for dataset in (dname, ):
        mode = {
            'er': 'static',
            'cl': 'static',
            'sbm': 'static',
            'graphrnn': 'incremental' if dataset == 'coauth-dblp' else 'static',
            'verg': 'static',
            'dyverg': 'incremental'
        }
        with open(join(rootpath, f'results/mmds/aggregate{dataset.replace("-", "").upper()}{statistic.replace("-", "").lower()}.mmd'), 'w') as outfile:
            for model in models:
                outfile.write(r'\pgfplotstableread{')
                outfile.write('\nts mmd\n')

                datatensor = {}
                with open(join(rootpath, f'results/mmds/{statistic}/{dataset}_{model}_{mode[model]}_{statistic}.mmd'), 'r') as infile:
                    next(infile)
                    for line in infile:
                        time, trial, mmd = line.strip().split(',')
                        datatensor[int(time)] = float(mmd)

                for t in datatensor:
                    if t == 0 or t > 10:
                        outfile.write(f'% {t} {datatensor[t]}\n')
                    else:
                        outfile.write(f'{t} {datatensor[t]}\n')

                outfile.write(fr'}}{{\{statistic.replace("-", "")}{dataset.replace("-", "").upper()}{model}}}')
                outfile.write('\n\n')

