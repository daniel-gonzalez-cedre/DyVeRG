# Email DNC temporal dataset
- data acquired from The Network Data Repository.
- `https://networkrepository.com/ia-enron-email-dynamic.php`

# quick stats
- First timestamp: ` `
- Last timestamp: ` `
- Number of interactions: `1_148_072`

# raw data
- Each line contains a sender, receiver, 1, and UNIX epoch timestamp (if valid) separated by spaces.

# processed
- UNIX epochs are then converted into datetime strings `year,month,day,hour,minute,second`.
- The strings are filtered to remove all but the `year` and `month` information.
- The lines are then relabelled with the new monthly timestamps.
