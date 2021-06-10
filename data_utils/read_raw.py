"""
From my Transformer's data.py
"""
import pandas as pd
import json
import random
from os import listdir
from os.path import isfile, join, isdir

from tqdm import tqdm

REAL, FAKE = 1, 0
SEED = 123
random.seed(SEED)

def read_politifact_input(dataset='politifact'):
    in_dir = f'/rwproject/kdd-db/20-rayw1/FakeNewsNet/code/fakenewsnet_dataset/{dataset}'
    rumorities = {'real': REAL, 'fake': FAKE}
    inpt = []
    for rumority, label in rumorities.items():
        for news_id in tqdm(listdir(join(in_dir, rumority)), desc=f'{dataset}-{rumority}'):
            content_fn = join(in_dir, rumority, news_id, 'news content.json')
            if not isfile(content_fn): continue
            with open(content_fn, 'r') as f:
                content = json.load(f)
            has_image = int(len(content["top_img"]) > 0)
            num_images = len(content["images"])
            num_exclam = (content["title"] + content["text"]).count("!")
            tp = join(in_dir, rumority, news_id, 'tweets')
            num_tweets = len(listdir(tp)) if isdir(tp) else 0
            rp = join(in_dir, rumority, news_id, 'retweets')
            num_retweets = len(listdir(rp)) if isdir(rp) else 0
            other_features = [has_image, num_images, num_exclam, num_tweets, num_retweets]
            inpt.append([news_id, content['title'] + " " + content["text"], label, other_features])
    return inpt

def read_pheme_input(in_dir = '/rwproject/kdd-db/20-rayw1/pheme-figshare'):
    rumorities = {'non-rumours': REAL, 'rumours': FAKE}
    inpt = []
    for event_raw in listdir(in_dir):
        if event_raw[-16:] != '-all-rnr-threads': continue
        # {event}-all-rnr-threads
        event = event_raw[:-16]
        for rumority, label in rumorities.items():
            for news_id in tqdm(listdir(join(in_dir, event_raw, rumority)), desc=f'pheme-{event}-{rumority}'):
                if news_id == '.DS_Store': continue
                tweets_dir = join(in_dir, event_raw, rumority, news_id, 'source-tweets')
                for tweets_fn in listdir(tweets_dir):
                    if tweets_fn == '.DS_Store': continue 
                    with open(join(tweets_dir, tweets_fn), 'r') as f:
                        tweet = json.load(f)
                        other_features = [
                            tweet["favorite_count"], tweet["retweet_count"], tweet['user']['followers_count'], 
                            tweet['user']['statuses_count'], tweet['user']['friends_count'], tweet['user']['favourites_count'],
                            len(tweet['user']['description'].split(' ')) if tweet['user']['description'] else 0,
                        ]
                        inpt.append([tweet["id_str"], tweet['text'], label, other_features])
    return inpt