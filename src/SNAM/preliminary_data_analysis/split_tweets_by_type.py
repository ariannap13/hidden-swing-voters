import pandas as pd
import json
import os
import bz2
from concurrent.futures import ThreadPoolExecutor
import threading

# Initialize a counter and a lock for thread-safe operations
counter = 0
lock = threading.Lock()

def process_file(file_path):
    """
    Function to process a file and extract tweets from VIPs.

    :param file_path: The path to the file to process.

    :return: A dictionary of tweets from VIPs.
    """
    local_dict_overall_tweet = {}
    local_dict_overall_retweet = {}
    local_dict_overall_quote = {}

    with bz2.open(file_path, "rt") as f:
        for line in f:
            try:
                tweet = json.loads(line)
                if 'author_id' in tweet:
                    author_id = tweet['author_id']
                elif 'user' in tweet:
                    author_id = tweet['user'].get('id_str', "")
                else:
                    continue

                tweet_id = tweet.get("id", "") or tweet.get("id_str", "")
                created_at = tweet.get("created_at", "")

                is_retweet = ("retweeted_status" in tweet and tweet["retweeted_status"]) or \
                                 ("referenced_tweets" in tweet and tweet["referenced_tweets"][0]["type"] == "retweeted")
                is_quote = ("referenced_tweets" in tweet and tweet["referenced_tweets"][0]["type"] == "quoted")

                if not is_retweet and not is_quote:
                    dict_tweet = {"author_id": author_id, "created_at": created_at}
                    local_dict_overall_tweet[tweet_id] = dict_tweet
                elif is_retweet:
                    dict_retweet = {"author_id": author_id, "created_at": created_at}
                    local_dict_overall_retweet[tweet_id] = dict_retweet
                elif is_quote:
                    dict_quote = {"author_id": author_id, "created_at": created_at}
                    local_dict_overall_quote[tweet_id] = dict_quote

            except json.JSONDecodeError:
                continue

    return local_dict_overall_tweet, local_dict_overall_retweet, local_dict_overall_quote

def update_progress(future):
    """
    Function to update the progress of the files processed.

    :param future: The future object to get the result from.

    :return: None
    """
    global counter
    local_dict_overall_tweet, local_dict_overall_retweet, local_dict_overall_quote = future.result()
    with lock:
        dict_overall_tweet.update(local_dict_overall_tweet)
        dict_overall_retweet.update(local_dict_overall_retweet)
        dict_overall_quote.update(local_dict_overall_quote)
        counter += 1
        print(f"Files processed: {counter}/{total_files}", flush=True)


# def update_progress(future):
#     """
#     Function to update the progress of the files processed.

#     :param future: The future object to get the result from.

#     :return: None
#     """
#     global counter
#     dict_overall.update(future.result())
#     with lock:
#         counter += 1
#         print(f"Files processed: {counter}/{total_files}", flush=True)

# Paths to data directories
data_dirs = [
    "../../../data/ita-2022_twitter_data/search_tweets/",
    "../../../data/ita-2022_twitter_data/tweets/"
]

dict_overall_tweet = {}
dict_overall_retweet = {}
dict_overall_quote = {}
total_files = sum(len(os.listdir(data_dir)) for data_dir in data_dirs)  # Total number of files to process
print(f"Total files to process: {total_files}", flush=True)

with ThreadPoolExecutor() as executor:
    futures = []
    for data_dir in data_dirs:
        files = os.listdir(data_dir)
        for file in files:
            file_path = os.path.join(data_dir, file)
            future = executor.submit(process_file, file_path)
            future.add_done_callback(update_progress)
            futures.append(future)

# No need to loop through futures to update dict_overall; done_callbacks handle it

# Save dict_overall to a file (json)
dict_dir = "../../../data/"
with open(os.path.join(dict_dir, "tweets_all.json"), "w") as f:
    json.dump(dict_overall_tweet, f)
with open(os.path.join(dict_dir, "retweets_all.json"), "w") as f:
    json.dump(dict_overall_retweet, f)
with open(os.path.join(dict_dir, "quote_all.json"), "w") as f:
    json.dump(dict_overall_quote, f)
