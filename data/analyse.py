def analyse(dataname, mode):
    assert dataname in ['email-dnc', 'email-enron', 'email-eucore', 'facebook-links']
    assert mode in ('r', 'raw', 'f', 'full', 'p', 'pruned')

    if mode in ('f', 'full'):  # full processed
        fname = f'{dataname}/{dataname}_full.edgelist'
        delim = ','
    elif mode in ('p', 'pruned'):  # pruned processed
        fname = f'{dataname}/{dataname}_pruned.edgelist'
        delim = ','
    else:  # raw
        if dataname == 'email-dnc':
            ext = 'edges'
            delim = ','
        elif dataname == 'email-enron':
            ext = 'csv'
            delim = ','
        elif dataname == 'email-eucore':
            ext = 'txt'
            delim = ' '
        elif dataname == 'facebook-links':
            ext = 'txt'
            delim = '\t'
        fname = f'{dataname}/{dataname}.{ext}'

    timestamps = set()
    nodes = set()
    edges = set()
    edges_t = set()
    edges_self = set()
    uinteractions = set()
    dinteractions = set()

    with open(fname, 'r') as infile:
        for line in infile:
            u, v, t = line.strip().split(delim)
            # if dataname == 'email-eucore':
            #     u, v, t = line.strip().split(delim)
            # elif dataname == 'email-enron':
            #     u, v, t = line.strip().split(delim)
            # else:
            #     pass

            if t in ('', r'\N'):
                continue

            timestamps |= {int(t)}
            nodes |= {u, v}
            edges |= {frozenset({u, v})}
            edges_t |= {(frozenset({u, v}), t)}
            if u == v:
                edges_self |= {frozenset({u, v})}
            uinteractions |= {(frozenset({u, v}), t)}
            dinteractions |= {(u, v, t)}

            # edges = [line.strip().split(',') for line in infile]

    timestamps = sorted(timestamps)

    print(f'first timestamp: {timestamps[0]}')
    print(f'last timestamp: {timestamps[-1]}')
    print(f'number of timestamps: {len(timestamps)}')
    print(f'number of nodes: {len(nodes)}')
    print(f'number of edges: {len(edges)}')
    print(f'number of self-edges: {len(edges_self)}')
    print(f'number of (undirected) interactions: {len(uinteractions)}')
    print(f'number of (directed) interactions: {len(dinteractions)}')

    if mode in ('f', 'full', 'p', 'pruned'):
        snapshots = {}
        for e, t in edges_t:
            if len(e) == 1:
                u, = e
                v, = e
            elif len(e) == 2:
                u, v = e
            if t in snapshots:
                snapshots[t] |= {frozenset({u, v})}
            else:
                snapshots[t] = {frozenset({u, v})}
        print(f'number of snapshots: {len(snapshots)}')
        print('\ttime: \t\torder:\tsize:')
        for t, e in sorted(snapshots.items()):
            nn = {u for n in e for u in n}
            print(f'\t{t}:   \t{len(nn)}\t{len(e)}')

    return


if __name__ == '__main__':
    dname = input('Enter name of dataset to analyse: ').lower().strip()
    pmode = input('Raw (r), full (f), or pruned (p)?: ').lower().strip()
    analyse(dname, pmode)
