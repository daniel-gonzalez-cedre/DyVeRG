# Email DNC temporal dataset
- data acquired from SNAP.
- `https://snap.stanford.edu/data/email-Eu-core-temporal.html`

# quick stats
- First timestamp: `1970-01-01 00:00:00`
- Last timestamp: `1972-03-14 22:14:14`
- Number of timestamps: `207_880`
- Number of graph snapshots: `19`

- Number of unique nodes: `986`
- Number of unique edges: `16_064`
- Number of self-edges: `0`

- Number of (undirected) interactions: `327_333`
- Number of (directed) interactions: `327_336`

# raw data
- Each line contains a sender, receiver, and UNIX epoch timestamp separated by spaces.
    - `sender receiver timestamp\n`

# processed
- UNIX epochs are then converted into datetime strings `year,month,day,hour,minute,second`.
- The strings are filtered to remove all but the `year` and `month` information.
- The lines are then relabelled with the new monthly timestamps.
