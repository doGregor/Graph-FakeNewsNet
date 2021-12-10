import pandas as pd
import numpy as np
import json
import pickle
import torch
from data_preprocessing.load_data import *
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
from data_preprocessing.feature_extraction import *


def create_homogeneous_graph(news_id_dict, dataset='politifact', include_tweets=True, include_users=True,
                             include_user_timeline_tweets=True, include_retweets=True, include_user_followers=True,
                             include_user_following=True, to_undirected=True):
    node_ids = {'article': [],
                'tweet': [],
                'user': []}
    node_ids_all = []
    graph = Data(x=[], edge_index=[[], []], y=[])
    for subset, news_ids in news_id_dict.items():
        for news_id in news_ids:
            if content_available(news_id=news_id, dataset=dataset, subset=subset):
                news_content = get_news_content(news_id=news_id, dataset=dataset, subset=subset)
                # news node feature
                graph.x.append(0)
                graph.y.append(0)
                node_ids['article'].append(news_id)
                node_ids_all.append(news_id)
                if include_tweets:
                    tweets_path, tweet_ids = get_news_tweet_ids(news_id=news_id, dataset=dataset, subset=subset)
                    for tweet in tweet_ids:
                        tweet_data = open_tweet_json(tweets_path, tweet)
                        if tweet_data['id'] not in node_ids['tweet']:
                            # tweets node features
                            graph.x.append(1)
                            graph.y.append(1)
                            node_ids['tweet'].append(tweet_data['id'])
                            node_ids_all.append(tweet_data['id'])
                        node_id_news = node_ids_all.index(news_id)
                        node_id_tweet = node_ids_all.index(tweet_data['id'])
                        graph.edge_index[0] += [node_id_news, node_id_tweet]
                        graph.edge_index[1] += [node_id_tweet, node_id_news]
                        if include_users:
                            user_information = get_user_information(tweet_data['user']['id'])
                            if user_information:
                                if user_information['id'] not in node_ids['user']:
                                    # user node features
                                    graph.x.append(2)
                                    graph.y.append(2)
                                    node_ids['user'].append(user_information['id'])
                                    node_ids_all.append(user_information['id'])
                                node_id_user = node_ids_all.index(user_information['id'])
                                graph.edge_index[0] += [node_id_tweet, node_id_user]
                                graph.edge_index[1] += [node_id_user, node_id_tweet]
                        #else:
                        #    print(f"[WARNING] excluding sample with id {news_id} no user available")
                        if include_retweets:
                            retweets_path, retweet_ids = get_retweet_ids(news_id=news_id, dataset=dataset, subset=subset)
                            if tweet in retweet_ids:
                                retweets_data = open_retweet_json(retweets_path, tweet)
                                for retweet in retweets_data:
                                    if retweet['id'] not in node_ids['tweet']:
                                        # retweets node features
                                        graph.x.append(1)
                                        graph.y.append(1)
                                        node_ids['tweet'].append(retweet['id'])
                                        node_ids_all.append(retweet['id'])
                                    node_id_retweet = node_ids_all.index(retweet['id'])
                                    graph.edge_index[0] += [node_id_retweet]
                                    graph.edge_index[1] += [node_id_tweet]
                                    if include_users:
                                        user_information = get_user_information(retweet['user']['id'])
                                        if user_information:
                                            if user_information['id'] not in node_ids['user']:
                                                # user node features
                                                graph.x.append(2)
                                                graph.y.append(2)
                                                node_ids['user'].append(user_information['id'])
                                                node_ids_all.append(user_information['id'])
                                            node_id_user = node_ids_all.index(user_information['id'])
                                            graph.edge_index[0] += [node_id_user]
                                            graph.edge_index[1] += [node_id_retweet]
            else:
                print(f"[WARNING] excluding sample with id {news_id} no news or tweets available")
    if include_users and include_user_timeline_tweets and len(node_ids['user']) > 0:
        for user_id in node_ids['user']:
            user_timeline_tweets = get_user_timeline_tweets(user_id)
            node_id_user = node_ids_all.index(user_id)
            if len(user_timeline_tweets) > 0:
                for user_timeline_tweet_data in user_timeline_tweets:
                    if user_timeline_tweet_data['id'] not in node_ids['tweet']:
                        # timeline tweets node features
                        graph.x.append(1)
                        graph.y.append(1)
                        node_ids['tweet'].append(user_timeline_tweet_data['id'])
                        node_ids_all.append(user_timeline_tweet_data['id'])
                    node_id_tweet = node_ids_all.index(user_timeline_tweet_data['id'])
                    graph.edge_index[0] += [node_id_user]
                    graph.edge_index[1] += [node_id_tweet]
            if include_user_followers:
                user_followers = get_user_followers(user_id)
                if len(user_followers) > 0:
                    for follower_id in user_followers:
                        if follower_id in node_ids['user']:
                            node_id_follower = node_ids_all.index(follower_id)
                            graph.edge_index[0] += [node_id_follower]
                            graph.edge_index[1] += [node_id_user]
                        elif get_user_information(follower_id):
                            user_information = get_user_information(follower_id)
                            # followers user features
                            graph.x.append(2)
                            graph.y.append(2)
                            node_ids['user'].append(user_information['id'])
                            node_ids_all.append(user_information['id'])
                            graph.edge_index[0] += [len(node_ids_all)-1]
                            graph.edge_index[1] += [node_id_user]
            if include_user_following:
                user_following = get_user_following(user_id)
                if len(user_following) > 0:
                    for following_id in user_following:
                        if following_id in node_ids['user']:
                            node_id_following = node_ids_all.index(following_id)
                            graph.edge_index[0] += [node_id_user]
                            graph.edge_index[1] += [node_id_following]
                        elif get_user_information(following_id):
                            user_information = get_user_information(following_id)
                            # following user features
                            graph.x.append(2)
                            graph.y.append(2)
                            node_ids['user'].append(user_information['id'])
                            node_ids_all.append(user_information['id'])
                            graph.edge_index[0] += [node_id_user]
                            graph.edge_index[1] += [len(node_ids_all)-1]
    graph.x = torch.tensor(graph.x, dtype=torch.float32)
    graph.y = torch.tensor(graph.y, dtype=torch.long)
    graph.edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
    graph.num_classes = torch.unique(graph.y).size()[0]
    graph = graph.coalesce()
    if to_undirected:
        graph = T.ToUndirected()(graph)
    return graph


def create_heterogeneous_graph(news_id_dict, dataset='politifact', include_tweets=True, include_users=True,
                               include_user_timeline_tweets=True, include_retweets=True, include_user_followers=True,
                               include_user_following=True, to_undirected=True, include_text=False):
    node_ids = {'article': [],
                'tweet': [],
                'user': []}
    graph = HeteroData()
    graph['article'].x = []
    graph['article'].y = []
    if include_tweets:
        graph['tweet'].x = [[], []]
        graph['tweet', 'cites', 'article'].edge_index = [[], []]
    if include_users:
        graph['user'].x = [[], []]
        graph['user', 'posts', 'tweet'].edge_index = [[], []]
    if include_user_followers or include_user_following:
        graph['user', 'follows', 'user'].edge_index = [[], []]
    if include_retweets:
        graph['tweet', 'retweets', 'tweet'].edge_index = [[], []]

    for subset, news_ids in news_id_dict.items():
        for news_id in news_ids:
            if content_available(news_id=news_id, dataset=dataset, subset=subset):
                news_content = get_news_content(news_id=news_id, dataset=dataset, subset=subset)
                # news node feature
                graph['article'].x.append(get_news_features(news_content))
                if subset == 'fake':
                    graph['article'].y.append(1)
                elif subset == 'real':
                    graph['article'].y.append(0)
                node_ids['article'].append(news_id)
                if include_tweets:
                    tweets_path, tweet_ids = get_news_tweet_ids(news_id=news_id, dataset=dataset, subset=subset)
                    for tweet in tweet_ids:
                        tweet_data = open_tweet_json(tweets_path, tweet)
                        if tweet_data['id'] not in node_ids['tweet']:
                            # tweets node features
                            graph['tweet'].x[0].append(get_tweet_features(tweet_data)[0])
                            graph['tweet'].x[1].append(get_tweet_features(tweet_data)[1:])
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
                                    graph['user'].x[0].append(get_user_features(user_information)[0])
                                    graph['user'].x[1].append(get_user_features(user_information)[1:])
                                    node_ids['user'].append(user_information['id'])
                                node_id_user = node_ids['user'].index(user_information['id'])
                                graph['user', 'posts', 'tweet'].edge_index[0] += [node_id_user]
                                graph['user', 'posts', 'tweet'].edge_index[1] += [node_id_tweet]
                        #else:
                        #    print(f"[WARNING] excluding sample with id {news_id} no user available")
                        if include_retweets:
                            retweets_path, retweet_ids = get_retweet_ids(news_id=news_id, dataset=dataset, subset=subset)
                            if tweet in retweet_ids:
                                retweets_data = open_retweet_json(retweets_path, tweet)
                                for retweet in retweets_data:
                                    if retweet['id'] not in node_ids['tweet']:
                                        # retweets node features
                                        graph['tweet'].x[0].append(get_tweet_features(retweet)[0])
                                        graph['tweet'].x[1].append(get_tweet_features(retweet)[1:])
                                        node_ids['tweet'].append(retweet['id'])
                                    node_id_retweet = node_ids['tweet'].index(retweet['id'])
                                    graph['tweet', 'retweets', 'tweet'].edge_index[0] += [node_id_retweet]
                                    graph['tweet', 'retweets', 'tweet'].edge_index[1] += [node_id_tweet]
                                    if include_users:
                                        user_information = get_user_information(retweet['user']['id'])
                                        if user_information:
                                            if user_information['id'] not in node_ids['user']:
                                                # user node features
                                                graph['user'].x[0].append(get_user_features(user_information)[0])
                                                graph['user'].x[1].append(get_user_features(user_information)[1:])
                                                node_ids['user'].append(user_information['id'])
                                            node_id_user = node_ids['user'].index(user_information['id'])
                                            graph['user', 'posts', 'tweet'].edge_index[0] += [node_id_user]
                                            graph['user', 'posts', 'tweet'].edge_index[1] += [node_id_retweet]
            else:
                print(f"[WARNING] excluding sample with id {news_id} no news or tweets available")
                if len(news_ids) == 1:
                    graph['article'].x = torch.tensor(graph['article'].x, dtype=torch.float32)
                    return graph
    if include_users and include_user_timeline_tweets and len(node_ids['user']) > 0:
        for user_id in node_ids['user']:
            user_timeline_tweets = get_user_timeline_tweets(user_id)
            node_id_user = node_ids['user'].index(user_id)
            if len(user_timeline_tweets) > 0:
                for user_timeline_tweet_data in user_timeline_tweets:
                    if user_timeline_tweet_data['id'] not in node_ids['tweet']:
                        graph['tweet'].x[0].append(get_tweet_features(user_timeline_tweet_data)[0])
                        graph['tweet'].x[1].append(get_tweet_features(user_timeline_tweet_data)[1:])
                        node_ids['tweet'].append(user_timeline_tweet_data['id'])
                    node_id_tweet = node_ids['tweet'].index(user_timeline_tweet_data['id'])
                    graph['user', 'posts', 'tweet'].edge_index[0] += [node_id_user]
                    graph['user', 'posts', 'tweet'].edge_index[1] += [node_id_tweet]
            if include_user_followers:
                user_followers = get_user_followers(user_id)
                if len(user_followers) > 0:
                    for follower_id in user_followers:
                        if follower_id in node_ids['user']:
                            node_id_follower = node_ids['user'].index(follower_id)
                            graph['user', 'follows', 'user'].edge_index[0] += [node_id_follower]
                            graph['user', 'follows', 'user'].edge_index[1] += [node_id_user]
                        elif get_user_information(follower_id):
                            user_information = get_user_information(follower_id)
                            graph['user'].x[0].append(get_user_features(user_information)[0])
                            graph['user'].x[1].append(get_user_features(user_information)[1:])
                            node_ids['user'].append(user_information['id'])
                            graph['user', 'follows', 'user'].edge_index[0] += [len(node_ids['user'])-1]
                            graph['user', 'follows', 'user'].edge_index[1] += [node_id_user]
            if include_user_following:
                user_following = get_user_following(user_id)
                if len(user_following) > 0:
                    for following_id in user_following:
                        if following_id in node_ids['user']:
                            node_id_following = node_ids['user'].index(following_id)
                            graph['user', 'follows', 'user'].edge_index[0] += [node_id_user]
                            graph['user', 'follows', 'user'].edge_index[1] += [node_id_following]
                        elif get_user_information(following_id):
                            user_information = get_user_information(following_id)
                            graph['user'].x[0].append(get_user_features(user_information)[0])
                            graph['user'].x[1].append(get_user_features(user_information)[1:])
                            node_ids['user'].append(user_information['id'])
                            graph['user', 'follows', 'user'].edge_index[0] += [node_id_user]
                            graph['user', 'follows', 'user'].edge_index[1] += [len(node_ids['user'])-1]

    graph['article'].x = torch.tensor(text_embeddings(graph['article'].x), dtype=torch.float32)
    graph['article'].y = torch.tensor(graph['article'].y, dtype=torch.long)
    if include_tweets:
        if include_text:
            graph['tweet'].x = torch.tensor(np.concatenate((text_embeddings(graph['tweet'].x[0]), np.asarray(graph['tweet'].x[1])), axis=1), dtype=torch.float32)
        else:
            graph['tweet'].x = torch.tensor(graph['tweet'].x[1], dtype=torch.float32)
        graph['tweet', 'cites', 'article'].edge_index = torch.tensor(graph['tweet', 'cites', 'article'].edge_index, dtype=torch.long)
    if include_users:
        if include_text:
            graph['user'].x = torch.tensor(np.concatenate((text_embeddings(graph['user'].x[0]), np.asarray(graph['user'].x[1])), axis=1), dtype=torch.float32)
        else:
            graph['user'].x = torch.tensor(graph['user'].x[1], dtype=torch.float32)
        graph['user', 'posts', 'tweet'].edge_index = torch.tensor(graph['user', 'posts', 'tweet'].edge_index, dtype=torch.long)
    if include_user_followers or include_user_following:
        graph['user', 'follows', 'user'].edge_index = torch.tensor(graph['user', 'follows', 'user'].edge_index, dtype=torch.long)
    if include_retweets:
        graph['tweet', 'retweets', 'tweet'].edge_index = torch.tensor(graph['tweet', 'retweets', 'tweet'].edge_index, dtype=torch.long)
    graph = graph.coalesce()
    if to_undirected:
        graph = T.ToUndirected(merge=False)(graph)
    return graph


def make_undirected(graph):
    for edge_relation in graph.metadata()[1]:
        print([graph[edge_relation]['edge_index'][1], graph[edge_relation]['edge_index'][0]])
        graph[edge_relation[2], 'rev_'+edge_relation[1], edge_relation[0]].edge_index = [graph[edge_relation]['edge_index'][1], graph[edge_relation]['edge_index'][0]]
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
    from data_preprocessing.visualization import *

    ids_true, ids_fake = get_news_ids()

    #for id in ids_fake[0:5]:

    graph = create_homogeneous_graph({'fake': list(ids_fake[:10])}, include_user_followers=False, include_user_following=False,
                                     to_undirected=True)
    print(graph)
    visualize_graph(graph, labels=True)
