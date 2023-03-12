import subprocess


# models = ('er', 'cl', 'sbm', 'verg', 'dyverg')
models = ('dyverg',)
datasets = ('email-dnc', 'email-enron', 'email-eucore', 'facebook-links', 'coauth-dblp')
modes = {
    'er': 'static',
    'cl': 'static',
    'sbm': 'static',
    'graphrnn': 'static',  # ???
    'verg': 'static',
    'dyverg': 'incremental'  # ???
}
statnames = ('spectrum',)
# statnames = ('degree-distribution',
#              'average-degree', 'triangle-count', 'clustering', 'transitivity')
for model in models:
    for dataset in datasets:
        for statname in statnames:
            subprocess.call(['python', 'experiments/comparison_mmd.py', model, modes[model], statname, dataset])
