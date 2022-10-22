from datetime import datetime as dt


def sieve(lines):
    return [(u, v, t) for u, v, t in lines if t != r'\N']


# (u, v, year, month, day, hour, minute, second)
def convert(filtered):
    return [(u, v, *dt.utcfromtimestamp(int(t)).strftime('%Y,%m,%d,%H,%M,%S').split(','))
            for u, v, t in filtered]


# split into monthly groups
def discretize(converted, granularity='monthly'):
    assert granularity in ['monthly', 'weekly']

    if granularity == 'monthly':
        discretized = [(u, v, int(str(year) + str(month)))
                       for u, v, year, month, _, _, _, _ in converted]
    elif granularity == 'weekly':
        yearmonthdays = [int(str(year) + str(month) + str(day)) % 7
                         for u, v, year, month, day, _, _, _ in converted]

        discretized = [(u, v, int(str(year) + str(month)))
                       for u, v, year, month, _, _, _, _ in converted]
    return sorted(discretized, key=lambda x: x[2])


def process(dataname):
    if dataname == 'facebook-links':
        ext = 'txt'
        sep = '\t'
    elif dataname == 'email-dnc':
        ext = 'edges'
        sep = ','
    elif dataname == 'email-eucore':
        ext = 'txt'
        sep = ' '
    else:
        raise NotImplementedError

    raw_name = f'{dataname}/{dataname}.{ext}'
    processed_name = f'{dataname}/{dataname}_processed.edgelist'

    print(f'Reading {dataname} from {raw_name}...', end=' ')

    with open(raw_name, 'r') as infile:
        lines = [line.strip().split(sep) for line in infile]

    print('done.')
    print(f'Processing {dataname}...', end=' ')

    filtered = sieve(lines)
    converted = convert(filtered)
    discretized = discretize(converted)

    print('done.')

    with open(processed_name, 'w') as outfile:
        for u, v, timestamp in discretized:
            outfile.write(f'{u},{v},{timestamp}\n')

    print(f'Successfully processed {dataname} and wrote to {processed_name}.')


if __name__ == '__main__':
    dataname = input('enter the name of the dataset to process: ')
    assert dataname in ['facebook-links', 'email-dnc', 'email-eucore']
    process(dataname)
