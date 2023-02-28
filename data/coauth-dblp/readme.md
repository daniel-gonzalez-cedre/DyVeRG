# Co-authorship network derived from DBLP papers
- data acquired from Austin Benson
- `https://www.cs.cornell.edu/~arb/data/coauth-DBLP/`

# quick stats
- First timestamp: `1938`
- Last timestamp: `2018`
- Number of timestamps: `77`
- Number of graph snapshots: `77`
- Effective snapshot indices: `28 -- 48`
- Effective snapshot timestamps: `1970 -- 1990`

- Number of unique nodes: ` `
- Number of unique edges: ` `
- Number of self-edges: `0`

- Number of (undirected) interactions: ` `
- Number of (directed) interactions: ` `


# raw data
- Each line contains a pair of authors and a four-digit year separated by commas.
    - `auth1,auth2,year\n`
- These are projected edges from the full simplicial data provided by Benson et al.
    - A hyperedge `v_1, ... v_n` at time `t` is projected down to a fully-connected `n`-clique,
      each of whose edges inherits the timestamp `t` from the hyperedge.


# processed
- Take edges with timestamps corresponding to years in the interval [1990, 2010] inclusive.
