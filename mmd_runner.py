import subprocess


models = ('er', 'cl', 'sbm', 'graphrnn', 'verg', 'dyverg')
datasets = ('email-dnc', 'email-enron', 'email-eucore', 'coauth-dblp', 'facebook-links')
# statnames = ('spectrum',)
statnames = ('degree-distribution', 'average-degree', 'triangle-count', 'clustering', 'transitivity')
# for model in models:
for _ in range(1):
    model = input('model: ').strip().lower()
    for dataset in datasets:
        modes = {
            'er': 'static',
            'cl': 'static',
            'sbm': 'static',
            'graphrnn': 'incremental' if dataset == 'coauth-dblp' else 'static',
            'verg': 'static',
            'dyverg': 'incremental'
        }
        for statname in statnames:
            subprocess.call(['python', 'experiments/comparison_mmd.py', model, modes[model], statname, dataset])
