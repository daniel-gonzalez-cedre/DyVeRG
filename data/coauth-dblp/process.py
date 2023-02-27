edges = []
with open('./coauth-dblp_full.edgelist', 'r') as infile:
    for line in infile:
        u, v, t = map(int, line.strip().split(','))
        if 1990 <= t <= 2010:
            edges.append((u, v, t))

with open('./coauth-dblp_pruned.edgelist', 'w') as outfile:
    for u, v, t in edges:
        outfile.write(f'{u},{v},{t}\n')
