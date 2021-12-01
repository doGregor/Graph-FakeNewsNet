import pandas as pd
import numpy as np
import json
import pickle
import torch
from data_preprocessing.load_data import *
from torch_geometric.data import Data, HeteroData


def create_homogeneous_graph(news_id_dict, dataset='politifact', include_tweets=True, include_users=True,
                             include_user_timeline_tweets=True, include_retweets=True, include_user_followers=True,
                             include_user_following=True, exclude_empty_samples=True):
    node_ids = []
    graph = Data(x=[], edge_index=[[], []], y=[])
    for subset, news_ids in news_id_dict.items():
        for news_id in news_ids:
            if content_available(news_id=news_id, dataset=dataset, subset=subset):
                news_content = get_news_content(news_id=news_id, dataset=dataset, subset=subset)
                # news node feature
                graph.x.append(0)
                graph.y.append(0)
                node_ids.append(news_id)
                if include_tweets:
                    tweets_path, tweet_ids = get_news_tweet_ids(news_id=news_id, dataset=dataset, subset=subset)
                    for tweet in tweet_ids:
                        tweet_data = open_tweet_json(tweets_path, tweet)
                        if tweet_data['id'] not in node_ids:
                            # tweets node features
                            graph.x.append(1)
                            graph.y.append(1)
                            node_ids.append(tweet_data['id'])
                        node_id_news = node_ids.index(news_id)
                        node_id_tweet = node_ids.index(tweet_data['id'])
                        graph.edge_index[0] += [node_id_news, node_id_tweet]
                        graph.edge_index[1] += [node_id_tweet, node_id_news]
                        if include_users:
                            user_information = get_user_information(tweet_data['user']['id'])
                            if user_information:
                                if user_information['id'] not in node_ids:
                                    # user node features
                                    graph.x.append(2)
                                    graph.y.append(2)
                                    node_ids.append(user_information['id'])
                                node_id_user = node_ids.index(user_information['id'])
                                graph.edge_index[0] += [node_id_tweet, node_id_user]
                                graph.edge_index[1] += [node_id_user, node_id_tweet]
                        else:
                            print(f"[WARNING] excluding sample with id {news_id} no user available")
            else:
                print(f"[WARNING] excluding sample with id {news_id} no news or tweets available")
    graph.x = torch.tensor(graph.x, dtype=torch.float32)
    graph.y = torch.tensor(graph.y, dtype=torch.long)
    graph.edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
    graph.num_classes = torch.unique(graph.y).size()[0]
    return graph


def create_heterogeneous_graph(news_id_dict, dataset='politifact', include_tweets=True, include_users=True,
                               include_user_timeline_tweets=True, include_retweets=True, include_user_followers=True,
                               include_user_following=True, exclude_empty_samples=True):
    node_ids = {'article': [],
                'tweet': [],
                'user': []}
    graph = HeteroData()
    graph['article'].x = []
    graph['article'].y = []
    if include_tweets:
        graph['tweet'].x = []
        graph['tweet', 'cites', 'article'].edge_index = [[], []]
    if include_users:
        graph['user'].x = []
        graph['user', 'posts', 'tweet'].edge_index = [[], []]

    for subset, news_ids in news_id_dict.items():
        for news_id in news_ids:
            if content_available(news_id=news_id, dataset=dataset, subset=subset):
                news_content = get_news_content(news_id=news_id, dataset=dataset, subset=subset)
                # news node feature
                graph['article'].x.append(0)
                graph['article'].y.append(0)
                node_ids['article'].append(news_id)
                if include_tweets:
                    tweets_path, tweet_ids = get_news_tweet_ids(news_id=news_id, dataset=dataset, subset=subset)
                    for tweet in tweet_ids:
                        tweet_data = open_tweet_json(tweets_path, tweet)
                        if tweet_data['id'] not in node_ids['tweet']:
                            # tweets node features
                            graph['tweet'].x.append(1)
                            node_ids['tweet'].append(tweet_data['id'])
                        node_id_news = node_ids['article'].index(news_id)
                        node_id_tweet = node_ids['tweet'].index(tweet_data['id'])
                        graph['tweet', 'cites', 'article'].edge_index[0] += [node_id_tweet]
                        graph['tweet', 'cites', 'article'].edge_index[1] += [node_id_news]
                        if include_users:
                            user_information = get_user_information(tweet_data['user']['id'])
                            if user_information:
                                if user_information['id'] not in node_ids['user']:
                                    # user node features
                                    graph['user'].x.append(2)
                                    node_ids['user'].append(user_information['id'])
                                node_id_user = node_ids['user'].index(user_information['id'])
                                graph['user', 'posts', 'tweet'].edge_index[0] += [node_id_user]
                                graph['user', 'posts', 'tweet'].edge_index[1] += [node_id_tweet]
                        else:
                            print(f"[WARNING] excluding sample with id {news_id} no user available")
            else:
                print(f"[WARNING] excluding sample with id {news_id} no news or tweets available")
    graph['article'].x = torch.tensor(graph['article'].x, dtype=torch.float32)
    graph['article'].y = torch.tensor(graph['article'].y, dtype=torch.long)
    if include_tweets:
        graph['tweet'].x = torch.tensor(graph['tweet'].x, dtype=torch.float32)
        graph['tweet', 'cites', 'article'].edge_index = torch.tensor(graph['tweet', 'cites', 'article'].edge_index, dtype=torch.long)
    if include_users:
        graph['user'].x = torch.tensor(graph['user'].x, dtype=torch.float32)
        graph['user', 'posts', 'tweet'].edge_index = torch.tensor(graph['user', 'posts', 'tweet'].edge_index, dtype=torch.long)
    return graph


def graph_to_pickle(graph, file_name):
    path = "../data/graphs/" + file_name + ".pickle"
    with open(path, 'wb') as handle:
        pickle.dump({'graph': graph}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def graph_from_pickle(file_name):
    path = "../data/graphs/" + file_name + ".pickle"
    with open(path, 'rb') as handle:
        return pickle.load(handle)


if __name__ == '__main__':
    ids_true, ids_fake = get_news_ids()

