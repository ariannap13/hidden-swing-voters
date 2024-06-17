import pandas as pd
import networkx as nx
import os
import bz2
import json
from datetime import datetime
import pprint
import pickle
import numpy as np

# paths to data directories, from Pierri et al. (2023)
data_dir_search_tweets = "../../../data/ita-2022_twitter_data/search_tweets/"
data_dir_tweets = "../../../data/ita-2022_twitter_data/tweets/"

# file available upon request
dict_dir = "../../../data/dict_tweet_byday/"

def get_author_retweeted(p_id, date_):
    """
    Get the author of the retweeted tweet.
    
    Parameters:
        p_id (str): The tweet ID.
        date_ (str): The date of the tweet.
    
    Returns:    
        str: The author of the retweeted tweet.
    """
    for dict_day in os.listdir(dict_dir):
        date_dict = dict_day.split("_")[1].split(".")[0]
        if datetime.strptime(date_dict, "%Y-%m-%d") <= datetime.strptime(date_, "%Y-%m-%d"):
            with open(dict_dir+"dict_"+date_dict+".pkl", 'rb') as handle:
                tweets_day = pickle.load(handle)
                if int(p_id) in tweets_day:
                    return tweets_day[int(p_id)]
    return None

# process files within data_dir_search_tweets
for file in os.listdir(data_dir_search_tweets):
    date_ = file.split("_")[0]
    file_edges = "file_edges_date_"+file.split('_')[0]
    with open("/home/arpe/FormaMentis/new_retweet_data/"+file_edges, "a+") as f:
        f.write("source,dest,text_tweet_id,created_at,type\n")
        df = pd.read_json(data_dir_search_tweets+file, lines=True)
        for i, row in df.iterrows():
            u = row["author_id"]
            if "in_reply_to_user_id" in row: 
                v = row["in_reply_to_user_id"]
                if not np.isnan(v):
                    towrite = f"{str(u)},{str(v)},{str(row['id'])},{row['created_at']},reply\n"
                    f.write(towrite)
            if type(row["entities"])==list:
                if "entities" in row and "mentions" in row["entities"]:
                    list_mentioned_id = [x["id"] for x in row["entities"]["mentions"]]
                    for v in list_mentioned_id:
                        towrite = f"{str(u)},{str(v)},{str(row['id'])},{row['created_at']},mention\n"
                        f.write(towrite)  
            if "referenced_tweets" in row: 
                if type(row["referenced_tweets"])!=list:
                    continue
                if len(row["referenced_tweets"]) > 0:
                    for el in row["referenced_tweets"]:
                        if el["type"] == "retweeted":
                            p_id = el["id"]
                            v = get_author_retweeted(p_id, date_)
                            if v is None:
                                continue
                            towrite = f"{str(u)},{str(v)},{str(p_id)},{row['created_at']},retweet\n"
                            f.write(towrite)     
                        elif el["type"] == "quoted":
                            p_id = el["id"]
                            v = get_author_retweeted(p_id, date_)
                            if v is None:
                                continue
                            towrite = f"{str(u)},{str(v)},{str(row['id'])},{row['created_at']},quote\n"
                            f.write(towrite)                            

# process files within data_dir_tweets
for file in os.listdir(data_dir_tweets):
    date_ = file.split("_")[0]
    if datetime.strptime(date_, "%Y-%m-%d") >= datetime.strptime("2022-09-02", "%Y-%m-%d"):
        file_edges = "file_edges_date_"+file.split('_')[0]
        with open("/home/arpe/FormaMentis/retweet_data_new/"+file_edges, "a+") as f:    
            if date_!="2022-09-02":
                f.write("source,dest,text_tweet_id,created_at,type\n")
            df = pd.read_json(data_dir_tweets+file, lines=True)
            for i, row in df.iterrows():
                if type(row["user"])==dict:
                    u = row["user"]["id"]
                    if "in_reply_to_user_id" in row: 
                        v = row["in_reply_to_user_id"]
                        if not np.isnan(v):
                            towrite = f"{str(u)},{str(v)},{str(row['id'])},{row['created_at']},reply\n"
                            f.write(towrite)
                    if type(row["entities"])==dict:
                        if "entities" in row and "user_mentions" in row["entities"]:
                            list_mentioned_id = [x["id"] for x in row["entities"]["user_mentions"]]
                            for v in list_mentioned_id:
                                towrite = f"{str(u)},{str(v)},{str(row['id'])},{row['created_at']},mention\n"
                                f.write(towrite)  
                    if "retweeted_status" in row: 
                        if type(row["retweeted_status"])==dict and row["retweeted_status"]["is_quote_status"]==False:
                            v = row["retweeted_status"]["user"]["id"]
                            towrite = f"{str(u)},{str(v)},{str(row['retweeted_status']['id'])},{row['created_at']},retweet\n"
                            f.write(towrite) 
                        elif (type(row["retweeted_status"])==dict and row["retweeted_status"]["is_quote_status"]==True):
                            v = row["retweeted_status"]["user"]["id"]
                            towrite = f"{str(u)},{str(v)},{str(row['id'])},{row['created_at']},quoted\n"
                            f.write(towrite)    
                        elif row["is_quote_status"]==True:
                            if type(row["quoted_status"])==dict:
                                v = row["quoted_status"]["user"]["id"]
                                towrite = f"{str(u)},{str(v)},{str(row['id'])},{row['created_at']},quoted\n"
                                f.write(towrite)                                
