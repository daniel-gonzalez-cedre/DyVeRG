from datetime import datetime as dt


def sieve(lines):
    if len(lines[0]) == 3:
        sieved = [(u, v, t) for u, v, t in lines if t != r'\N']
    elif len(lines[0]) == 4:
        sieved = [(u, v, t) for u, v, _, t in lines if t != r'\N']
    else:
        raise AssertionError
    return sieved


# (u, v, year, month, day, hour, minute, second)
def convert(filtered):
    return [(u, v, *dt.utcfromtimestamp(int(t)).strftime('%Y,%m,%d,%H,%M,%S').split(','))
            for u, v, t in filtered]


# split into monthly groups
def discretize(converted, granularity):
    assert granularity in ['monthly', 'weekly', 'monthly', 'hourly', 'minutely']

    if granularity == 'monthly':
        discretized = [(u, v, int(str(year) + str(month)))
                       for u, v, year, month, _, _, _, _ in converted]
    elif granularity == 'weekly':
        raise NotImplementedError
        # yearmonthdays = [int(str(year) + str(month) + str(day)) % 7
        #                  for u, v, year, month, day, _, _, _ in converted]

        # discretized = [(u, v, int(str(year) + str(month)))
        #                for u, v, year, month, _, _, _, _ in converted]
    elif granularity == 'hourly':
        discretized = [(u, v, int(str(year) + str(month) + str(day) + str(hour)))
                       for u, v, year, month, day, hour, _, _ in converted]
    elif granularity == 'minutely':
        discretized = [(u, v, int(str(year) + str(month) + str(day) + str(hour) + str(minute)))
                       for u, v, year, month, day, hour, minute, _ in converted]
    else:
        raise NotImplementedError
    return sorted(discretized, key=lambda x: x[2])


def process(dataname):
    if dataname == 'facebook-links':
        ext = 'txt'
        sep = '\t'
        granularity = 'monthly'
    elif dataname == 'email-dnc':
        ext = 'edges'
        sep = ','
        granularity = 'hourly'
    elif dataname == 'email-enron':
        ext = 'edges'
        sep = ' '
        granularity = 'minutely'
    elif dataname == 'email-eucore':
        ext = 'txt'
        sep = ' '
        granularity = 'monthly'
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
    discretized = discretize(converted, granularity)

    print('done.')

    with open(processed_name, 'w') as outfile:
        for u, v, timestamp in discretized:
            outfile.write(f'{u},{v},{timestamp}\n')

    print(f'Successfully processed {dataname} and wrote to {processed_name}.')


if __name__ == '__main__':
    name = input('enter the name of the dataset to process: ')
    assert name in ['facebook-links', 'email-dnc', 'email-enron', 'email-eucore']
    process(name)
