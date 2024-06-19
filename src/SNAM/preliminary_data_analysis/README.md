# Propaganda analysis

This folder contains code to perform some basic analysis and compute statistics on the tweets.

In particular:
- `split_tweets_by_type.py` creates three separate files for tweets, retweets and quotes to facilitate counting.
- `fix_dates_files.py` is a utility script that standardizes the date format of tweet json files to *{YYYY}-{MM}-{DD}*.
- `tweet_by_party.py` associates tweets to parties and produces a file used for counting purposes.
- `tweet_counts.ipynb` performs general statistical analysis on the data.

Note that all json files created should be passed through the script to `fix_dates.py` uniform the date format before being used.