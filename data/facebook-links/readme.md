# Facebook friendship links temporal dataset
- data acquired from Prof. Alan Mislove at the Max Planck Institute
- `https://socialnetworks.mpi-sws.org/data-wosn2009.html`

# quick stats
- First timestamp: `2006-09-05 11:15:29`
- Last timestamp: `2009-01-21 22:15:25`
- Number of timestamps: `736_674`
- Number of graph snapshots: `29`

- Number of unique nodes: `61_096`
- Number of unique edges: `614_797`
- Number of self-edges: `0`

- Number of (undirected) interactions: `788_135`
- Number of (directed) interactions: `905_565`

# raw data
- Each line contains a sender, receiver, and UNIX epoch timestamp (if valid) separated by tabs.

# processed
- Each line in the raw data is first filtered to remove `\N` timestamps
    - These lines corresponded to invalid timestamp data.
- UNIX epochs are then converted into datetime strings `year,month,day,hour,minute,second`.
- The strings are filtered to remove all but the `year` and `month` information.
- The lines are then relabelled with the new monthly timestamps.
