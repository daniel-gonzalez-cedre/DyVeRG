from tqdm import tqdm

edges: list[tuple[int, int, int]] = []

with open('coauth-DBLP-nverts.txt', 'r') as nvertsfile, \
     open('coauth-DBLP-times.txt', 'r') as timesfile, \
     open('coauth-DBLP-simplices.txt', 'r') as simplicesfile:
    nverts = [int(line.strip()) for line in nvertsfile]
    times = [str(line.strip()) for line in timesfile]
    simplices = [int(line.strip()) for line in simplicesfile]
    assert len(nverts) == len(times)

ctr = 0
for n, t in tqdm(zip(nverts, times), total=len(nverts)):
    nodes = simplices[ctr:ctr + n]
    edges += {frozenset({u, v, t})
              for u in simplices[ctr:ctr + n]
              for v in simplices[ctr:ctr + n]
              if u != v}
    ctr += n

sortie = lambda x, y, z: (x, y, z) if isinstance(z, str) else ((z, x, y) if isinstance(y, str) else (y, z, x))
sorted_edges = sorted([sortie(x, y, z) for x, y, z in edges], key=lambda a: a[2])

with open('./coauth-dblp_full.edgelist', 'w') as outfile:
    for u, v, t in sorted_edges:
        outfile.write(f'{u},{v},{t}\n')
