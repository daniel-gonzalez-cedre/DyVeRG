# Democratic National Committee email leak temporal dataset
- data acquired from The Network Data Repository
- `https://networkrepository.com/email-dnc.php`

# quick stats
- First timestamp: `2013-09-16 02:30:33`
- Last timestamp: `2016-05-25 09:26:28`
- Number of timestamps: `19_389`
- Number of graph snapshots: `11`

- Number of unique nodes: `1891`
- Number of unique edges: `4465`
- Number of self-edges: `81`

- Number of (undirected) interactions: `32_878`
- Number of (directed) interactions: `32_880`

# raw data
- Each line contains a sender, receiver, and UNIX epoch timestamp (if valid) separated by commas.

# processed
- UNIX epochs are then converted into datetime strings `year,month,day,hour,minute,second`.
- The strings are filtered to remove all but the `year` and `month` information.
- The lines are then relabelled with the new monthly timestamps.
