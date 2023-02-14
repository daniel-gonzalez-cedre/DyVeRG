# DOJ Enron email temporal dataset
- data acquired from Johns Hopkins University
- `https://www.cis.jhu.edu/~parky/Enron/enron.html`

# quick stats
- First timestamp: `1979-12-31 21:00:00`
- Last timestamp: `2002-06-21 19:40:19`
- Number of timestamps: `22_633`
- Number of graph snapshots: `31`

- Number of unique nodes: `184`
- Number of unique edges: `2216`
- Number of self-edges: `119`

- Number of (undirected) interactions: `38_172`
- Number of (directed) interactions: `38_184`

# raw data
- Each line contains a UNIX epoch timestamp, sender, and receiver separated by spaces.
    - `timestamp sender receiver\n`

# processed
- UNIX epochs are then converted into datetime strings `year,month,day,hour,minute,second`.
- The strings are filtered to remove all but the `year`, `month`, and `week` information.
- The lines are then relabelled with the new weekly timestamps.
