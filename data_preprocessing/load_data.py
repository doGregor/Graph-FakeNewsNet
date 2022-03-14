import pandas as pd
import numpy as np
import sys
import itertools
import os
import json


DATA_PATH = {
    'gossipcop_fake_dataset': '../data/dataset/gossipcop_fake.csv',
    'gossipcop_real_dataset': '../data/dataset/gossipcop_real.csv',
    'politifact_fake_dataset': '../data/dataset/politifact_fake.csv',
    'politifact_real_dataset': '../data/dataset/politifact_real.csv',
    'gossipcop_dir': '../data/fakenewsnet_dataset/gossipcop/',
    'politifact_dir': '../data/fakenewsnet_dataset/politifact/',
    'user_profiles_dir': '../data/fakenewsnet_dataset/user_profiles/',
    'user_timeline_tweets_dir': '../data/fakenewsnet_dataset/user_timeline_tweets/',
    'user_followers_dir': '../data/fakenewsnet_dataset/user_followers/',
    'user_following_dir': '../data/fakenewsnet_dataset/user_following/'
}


def __get_directories(dataset_name):
    if dataset_name == 'politifact':
        true = DATA_PATH['politifact_real_dataset']
        fake = DATA_PATH['politifact_fake_dataset']
        directory = DATA_PATH['politifact_dir']
    elif dataset_name == 'gossipcop':
        true = DATA_PATH['gossipcop_real_dataset']
        fake = DATA_PATH['gossipcop_fake_dataset']
        directory = DATA_PATH['gossipcop_dir']
    else:
        print("[ERROR] Wrong dataset parameter specified.")
        sys.exit(0)
    return true, fake, directory


def get_dataset_info(dataset='politifact'):
    true, fake, directory = __get_directories(dataset_name=dataset)
    true_data = pd.read_csv(true, sep=',')
    true_tweet_information = true_data['tweet_ids'].to_list()
    true_tweet_information = [str(x).split('\t') for x in true_tweet_information]
    print("True news samples:", len(true_tweet_information), "\t Number of related tweets:",
          len(list(itertools.chain.from_iterable(true_tweet_information))))
    fake_data = pd.read_csv(fake, sep=',')
    fake_tweet_information = fake_data['tweet_ids'].to_list()
    fake_tweet_information = [str(x).split('\t') for x in fake_tweet_information]
    print("Fake news samples:", len(fake_tweet_information), "\t Number of related tweets:",
          len(list(itertools.chain.from_iterable(fake_tweet_information))))


def get_news_ids(dataset='politifact'):
    true, fake, directory = __get_directories(dataset_name=dataset)
    return pd.read_csv(true, sep=',')['id'].to_numpy(), pd.read_csv(fake, sep=',')['id'].to_numpy()


def get_news_tweet_ids(news_id, dataset='politifact', subset='fake'):
    data_path = '../data/fakenewsnet_dataset/' + dataset + '/' + subset + '/' + news_id + '/tweets/'
    if os.path.exists(data_path):
        news_tweet_files = os.listdir(data_path)
        return (data_path, news_tweet_files)
    else:
        return ("", [])


def get_retweet_ids(news_id, dataset='politifact', subset='fake'):
    data_path = '../data/fakenewsnet_dataset/' + dataset + '/' + subset + '/' + news_id + '/retweets/'
    if os.path.exists(data_path):
        retweet_files = os.listdir(data_path)
        return (data_path, retweet_files)
    else:
        return ("", [])


def open_tweet_json(data_path, file_name):
    file_path = data_path + file_name
    if os.path.exists(file_path):
        with open(file_path, 'r') as tweet_json:
            tweet_data = json.load(tweet_json)
        return tweet_data
    else:
        return {}


def open_retweet_json(data_path, file_name):
    file_path = data_path + file_name
    if os.path.exists(file_path):
        with open(file_path, 'r') as retweet_json:
            retweet_data = json.load(retweet_json)
        return retweet_data['retweets']
    else:
        return []


def get_news_content(news_id, dataset='politifact', subset='fake'):
    file_path = '../data/fakenewsnet_dataset/' + dataset + '/' + subset + '/' + news_id + '/news content.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as news_json:
            news_data = json.load(news_json)
        return news_data
    else:
        return {}


def get_user_information(user_id):
    file_path = DATA_PATH['user_profiles_dir'] + str(user_id) + '.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as user_information_json:
            user_information = json.load(user_information_json)
        return user_information
    else:
        return {}


def get_user_timeline_tweets(user_id, n=5):
    file_path = DATA_PATH['user_timeline_tweets_dir'] + str(user_id) + '.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as user_timeline_tweets_json:
            user_timeline_tweets = json.load(user_timeline_tweets_json)
        if n == 0:
            return user_timeline_tweets
        else:
            return user_timeline_tweets[:n]
    else:
        return []


def get_user_followers(user_id):
    file_path = DATA_PATH['user_followers_dir'] + str(user_id) + '.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as user_followers_json:
            user_followers_info = json.load(user_followers_json)
        return user_followers_info['followers']
    else:
        return []


def get_user_following(user_id):
    file_path = DATA_PATH['user_following_dir'] + str(user_id) + '.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as user_following_json:
            user_following_info = json.load(user_following_json)
        return user_following_info['following']
    else:
        return []


def content_available(news_id, dataset='politifact', subset='fake'):
    if get_news_content(news_id=news_id, dataset=dataset, subset=subset) and (len(get_news_tweet_ids(news_id=news_id, dataset=dataset, subset=subset)[1]) > 0):
        news_content = get_news_content(news_id=news_id, dataset=dataset, subset=subset)
        if news_content['text'] != '':
            return True
        else:
            return False
    else:
        return False


if __name__ == '__main__':
    true, fake = get_news_ids()
    print(fake[0])

    print("NEWS CONTENT", get_news_content(fake[1]))
    data_path, tweet_ids = get_news_tweet_ids(fake[0])
    print("DATA PATH", data_path)
    print("TWEET IDS", tweet_ids)
    print("TWEET DATA", open_tweet_json(data_path, tweet_ids[0]))
    example_user = open_tweet_json(data_path, tweet_ids[0])['user']['id']
    print("USER DATA", get_user_information(example_user))
    print("USER TIMELINE TWEETS", get_user_timeline_tweets(41)[0])
