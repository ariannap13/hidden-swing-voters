import pandas as pd
import json
import os
import bz2
from concurrent.futures import ThreadPoolExecutor
import threading

# Initialize a counter and a lock for thread-safe operations
counter = 0
lock = threading.Lock()

def process_file(file_path, ids_set):
    local_dict_overall = {}
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
                text = tweet.get("text", "")

                if author_id in ids_set:
                    is_retweet = ("retweeted_status" in tweet and tweet["retweeted_status"]) or \
                                 ("referenced_tweets" in tweet and tweet["referenced_tweets"][0]["type"] == "retweeted")
                    if not is_retweet:
                        dict_tweet = {"author_id": author_id, "created_at": created_at, "text": text}
                        local_dict_overall[tweet_id] = dict_tweet
            except json.JSONDecodeError:
                continue
    return local_dict_overall

def update_progress(future):
    global counter
    dict_overall.update(future.result())
    with lock:
        counter += 1
        print(f"Files processed: {counter}/{total_files}", flush=True)

# Load the data (twitter handles of VIPs)
vips_info = pd.read_csv('../twitter_representatives_handles_final.csv')
ids_set = set(vips_info['ids'].values)
print(f"Number of VIPs: {len(ids_set)}", flush=True)

# Paths to data directories
data_dirs = [
    "../ita-2022_twitter_data/search_tweets/",
    "../ita-2022_twitter_data/tweets/"
]

dict_overall = {}
total_files = sum(len(os.listdir(data_dir)) for data_dir in data_dirs)  # Total number of files to process
print(f"Total files to process: {total_files}", flush=True)

with ThreadPoolExecutor() as executor:
    futures = []
    for data_dir in data_dirs:
        files = os.listdir(data_dir)
        for file in files:
            file_path = os.path.join(data_dir, file)
            future = executor.submit(process_file, file_path, ids_set)
            future.add_done_callback(update_progress)
            futures.append(future)

# No need to loop through futures to update dict_overall; done_callbacks handle it

# Save dict_overall to a file (json)
dict_dir = "./"
with open(os.path.join(dict_dir, "tweets_vips.json"), "w") as f:
    json.dump(dict_overall, f)
