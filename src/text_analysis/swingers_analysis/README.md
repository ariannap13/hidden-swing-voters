# Swingers analysis

This folder contains scripts and notebooks to perform the analysis of the vulnerability of swingers to different propaganda detection techniques.
In particular:
- `analyze_annotated_tweets.ipynb` performs an analysis of propaganda tweets over time, specifically in terms of propaganda techniques used by period and by party, considering an average baseline over the whole set of parties.
- `swingtype_analysis.ipynb` classifies swingers in hard swingers, soft swingers, non-real swingers, swingers from apolitical to political communities and swingers from political to apolitical communities.
- `get_swingers_vulnerability.py` produces a file that, for each tweet ID, saves the swinger user ID that retweeted it, the reference swing period and the date of retweet.
- `get_nonswingers_vulnerability.py` produces a file that, for each tweet ID, saves the non-swinger user ID that retweeted it, the reference swing period and the date of retweet. Non-swingers for a given swing period are defined as sample of users that do not swing during that period.
- `fix_dates_files.py` is a utility script that standardizes the date format of tweet json files to *{YYYY}-{MM}-{DD}*.
- `analyze_swingers.ipynb` performs some analyses on the swingers vulnerability. In particular, it measures the volume of vulnerable swingers in each swing period, it explores the techniques characterized by the highest vulnerability in each swing period and it analyses such vulnerability to techniques for different swingers types.
- `analyze_nonswingers.ipynb` performs some analyses on the non-swingers vulnerability. In particular, it measures the volume of vulnerable non-swingers in each swing period and it explores the techniques characterized by the highest vulnerability in each swing period.