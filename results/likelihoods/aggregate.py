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


if __name__ == '__main__':
    rootpath = git.Repo(getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
    models = ['er', 'cl', 'sbm', 'graphrnn', 'verg', 'dyverg']
    datasets = ['email-dnc', 'email-enron', 'email-eucore', 'facebook-links', 'coauth-dblp']

    for dataset in tqdm(datasets):
        mode = {
            'er': 'static',
            'cl': 'static',
            'sbm': 'static',
            'graphrnn': 'incremental' if dataset == 'coauth-dblp' else 'static',
            'verg': 'static',
            'dyverg': 'incremental'
        }
        with open(join(rootpath, f'results/likelihoods/aggregate{dataset.replace("-", "").upper()}.ll_ranks'), 'w') as outfile:
            means = {
                'er': [],
                'cl': [],
                'sbm': [],
                'graphrnn': [],
                'verg': [],
                'dyverg': []
            }
            cis = {
                'er': [],
                'cl': [],
                'sbm': [],
                'graphrnn': [],
                'verg': [],
                'dyverg': []
            }
            ranks = {
                'er': [],
                'cl': [],
                'sbm': [],
                'graphrnn': [],
                'verg': [],
                'dyverg': []
            }
            for model in models:
                datatensor = {}
                with open(join(rootpath, f'results/likelihoods/{dataset}_{model}_{mode[model]}_False.ll'), 'r') as infile:
                    next(infile)
                    for line in infile:
                        time, trial, ll = line.strip().split(',')
                        time, trial, ll = (int(time), int(trial), float(ll))
                        ll = -ll

                        if time in datatensor:
                            datatensor[time].append(ll)
                        else:
                            datatensor[time] = [ll]

                # for t in datatensor:
                for t in range(1, len(datatensor)):
                    try:
                        means[model].append(np.mean(datatensor[t]))
                        cis[model].append(confidence(datatensor[t]))
                    except:
                        print(t, model, datatensor)
                        exit()
                    # if t == 0 or t > 10:
                    #     outfile.write(f'% {t} {mean} {ci}\n')
                    # else:
                    #     outfile.write(f'{t} {mean} {ci}\n')

                # outfile.write(fr'}}{{\ll{dataset.replace("-", "").upper()}{model}}}')
                # outfile.write('\n\n')

            for t in range(0, 11):
                meanslice = [(mm, means[mm][t]) for mm in models if t < len(means[mm])]
                meanslice = sorted(meanslice, key=lambda x: x[1])
                modelslice = [mm for mm, _ in meanslice]
                for model in models:
                    try:
                        rank = modelslice.index(model)
                    except ValueError:
                        rank = 5
                    ranks[model].append(rank + 1)

            for model in models:
                outfile.write(r'\pgfplotstableread{')
                outfile.write('\nts avg ci rank\n')

                for t in range(0, 12):
                    try:
                        if t + 1 > 10:
                            outfile.write(f'% {t + 1} {means[model][t]} {cis[model][t]} {ranks[model][t]}\n')
                        else:
                            outfile.write(f'{t + 1} {means[model][t]} {cis[model][t]} {ranks[model][t]}\n')
                    except IndexError:
                        continue

                outfile.write(fr'}}{{\ll{dataset.replace("-", "").upper()}{model}}}')
                outfile.write('\n\n')
