import json
import pandas as pd

start_before = '2022-07-01'
start_during = '2022-08-26'
start_after = '2022-09-25'
end_after = '2022-10-31'

# Load the data
with open('../../data/tweets_vips.json') as f:
    tweets_vips = json.load(f)

representatives_handles = pd.read_csv('../../data/twitter_representatives_handles_final.csv')

for vip in tweets_vips:
    tweets_vips[vip]["party"] = representatives_handles[representatives_handles['ids'] == tweets_vips[vip]["author_id"]]['Party'].values[0]

# save json file
with open('../../data/tweets_vips_party.json', 'w') as f:
    json.dump(tweets_vips, f)
