# Email DNC temporal dataset
- data acquired from The Network Data Repository.
- `https://networkrepository.com/email-dnc.php`

# quick stats
- First timestamp: `2013-09-16 02:30:33`
- Last timestamp: `2016-05-25 09:26:28`
- Total number of months: `18`
- Number of interactions: `39_264`

# raw data
- Each line contains a sender, receiver, and UNIX epoch timestamp (if valid) separated by commas.

# processed
- UNIX epochs are then converted into datetime strings `year,month,day,hour,minute,second`.
- The strings are filtered to remove all but the `year` and `month` information.
- The lines are then relabelled with the new monthly timestamps.
