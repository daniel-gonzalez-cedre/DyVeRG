
def analyse(dataname):
    assert dataname in ['email-dnc', 'email-eucore', 'facebook-links']

    with open(f'{dataname}/{dataname}_processed.edgelist', 'r') as infile:
        edges = [line.strip().split(',') for line in infile]

    edge_dict = {}
    for u, v, t in edges:
        if t in edge_dict:
            edge_dict[t] += [(u, v)]
        else:
            edge_dict[t] = [(u, v)]

    print(len(edge_dict))

    return


if __name__ == '__main__':
    dataname = input('Enter name of dataset to analyse: ')
    analyse(dataname)
