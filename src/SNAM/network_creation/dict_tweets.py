import pandas as pd
import os
import pickle

data_dir = "../../../data/ita-2022_twitter_data/search_tweets/"

for file in os.listdir(data_dir):
    date = file.split("_")[0]
    dict_day = {}
    df = pd.read_json(data_dir+file, lines=True)
    for i, row in df.iterrows():
        # not (A and B) = not A or not B
        # this means that we consider a tweet only if it has at least one retweet or quote
        if not (row["public_metrics"]["quote_count"]==0 & row["public_metrics"]["retweet_count"]==0):
            user = row["author_id"]
            tweet_id = row["id"]
            dict_day[tweet_id] = user
    with open("../../../data/dict_tweet_byday/"+"dict_"+date+".pkl", 'wb') as handle:
        pickle.dump(dict_day, handle, protocol=pickle.HIGHEST_PROTOCOL)

