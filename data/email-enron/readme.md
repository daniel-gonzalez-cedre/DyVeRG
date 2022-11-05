# Email DNC temporal dataset
- data acquired from The Network Data Repository.
- `https://networkrepository.com/ia-enron-email-dynamic.php`

# quick stats
- First timestamp: `1970-01-03 07:00:01`
- Last timestamp: `1970-01-03 08:13:25`
- Number of interactions: `1_148_072`

# raw data
- Each line contains a sender, receiver, 1, and UNIX epoch timestamp (if valid) separated by spaces.

# processed
- UNIX epochs are then converted into datetime strings `year,month,day,hour,minute,second`.
- The strings are filtered to remove all but the `year`, `month`, `day`, `hour`, and `minute` information.
- The lines are then relabelled with the new minutely timestamps.
