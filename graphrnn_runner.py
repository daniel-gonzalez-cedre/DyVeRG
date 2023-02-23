import torch
torch.cuda.set_device(0)

from baselines.graphrnn.fit import fit
from baselines.graphrnn.gen import gen
from src.data import load_data
# from src.utils import mkdir

loaded = load_data('facebook-links')
graphs = [g for _, g in loaded]

for graph in graphs[2:]:
    args, model, output = fit(10*[graph], nn='rnn')
    generated_graphs = gen(args=args, model=model, output=output)
    print('!!!!!!!!!!!!!!!!')
    exit()
    for g in generated_graphs:
        print('\t', g.order(), g.size())

# args, model, output = fit(graphs)
# generated_graphs = gen(args=args, model=model, output=output)
# 
# print(len(graphs))
# for graph in generated_graphs:
#     print('!!!!!!!!!!!!!!!!', g.order(), g.size())
