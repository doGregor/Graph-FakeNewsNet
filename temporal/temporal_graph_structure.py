import pickle
from data_preprocessing.load_data import *
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
from datetime import datetime, timedelta
from data_preprocessing.feature_extraction import *
import torch



def create_heterogeneous_snapshot(news_id_dict, tweet_id_list, start_time, end_time,
                                  dataset='politifact', include_tweets=True,
                                  include_users=True, include_user_timeline_tweets=True, include_retweets=True,
                                  include_user_followers=True, include_user_following=True, add_new_users=False,
                                  to_undirected=True, include_text=False):
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

    for subset, news_id in news_id_dict.items():
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
                tweets_path, tweet_list = get_news_tweet_ids(news_id=news_id, dataset=dataset, subset=subset)
                for tweet in tweet_list:
                    tweet_data = open_tweet_json(tweets_path, tweet)
                    if tweet_data['id'] not in node_ids['tweet']:
                        # tweets node features
                        if tweet in tweet_id_list:
                            graph['tweet'].x[0].append(get_tweet_features(tweet_data)[0])
                            graph['tweet'].x[1].append(get_tweet_features(tweet_data)[1:])
                        else:
                            graph['tweet'].x[0].append('')
                            graph['tweet'].x[1].append([0]*2)
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
                                if tweet in tweet_id_list:
                                    graph['user'].x[0].append(get_user_features(user_information)[0])
                                    graph['user'].x[1].append(get_user_features(user_information)[1:])
                                else:
                                    graph['user'].x[0].append('')
                                    graph['user'].x[1].append([0]*4)
                                node_ids['user'].append(user_information['id'])
                            node_id_user = node_ids['user'].index(user_information['id'])
                            graph['user', 'posts', 'tweet'].edge_index[0] += [node_id_user]
                            graph['user', 'posts', 'tweet'].edge_index[1] += [node_id_tweet]
                    if include_retweets:
                        retweets_path, retweet_ids = get_retweet_ids(news_id=news_id, dataset=dataset, subset=subset)
                        if tweet in retweet_ids:
                            retweets_data = open_retweet_json(retweets_path, tweet)
                            for retweet in retweets_data:
                                if retweet['id'] not in node_ids['tweet']:
                                    if start_time <= datetime.strptime(retweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y') < end_time:
                                        # retweets node features
                                        graph['tweet'].x[0].append(get_tweet_features(retweet)[0])
                                        graph['tweet'].x[1].append(get_tweet_features(retweet)[1:])
                                    else:
                                        graph['tweet'].x[0].append('')
                                        graph['tweet'].x[1].append([0]*2)
                                    node_ids['tweet'].append(retweet['id'])
                                node_id_retweet = node_ids['tweet'].index(retweet['id'])
                                graph['tweet', 'retweets', 'tweet'].edge_index[0] += [node_id_retweet]
                                graph['tweet', 'retweets', 'tweet'].edge_index[1] += [node_id_tweet]
                                if include_users:
                                    user_information = get_user_information(retweet['user']['id'])
                                    if user_information:
                                        if user_information['id'] not in node_ids['user']:
                                            # user node features
                                            if tweet in tweet_id_list:
                                                graph['user'].x[0].append(get_user_features(user_information)[0])
                                                graph['user'].x[1].append(get_user_features(user_information)[1:])
                                            else:
                                                graph['user'].x[0].append('')
                                                graph['user'].x[1].append([0]*4)
                                            node_ids['user'].append(user_information['id'])
                                        node_id_user = node_ids['user'].index(user_information['id'])
                                        graph['user', 'posts', 'tweet'].edge_index[0] += [node_id_user]
                                        graph['user', 'posts', 'tweet'].edge_index[1] += [node_id_retweet]
        else:
            print(f"[WARNING] excluding sample with id {news_id} no news or tweets available")
            graph['article'].x = torch.tensor(graph['article'].x, dtype=torch.float32)
            return graph
    if include_users and include_user_timeline_tweets and len(node_ids['user']) > 0:
        for user_id in node_ids['user']:
            user_timeline_tweets = get_user_timeline_tweets(user_id, n=0)
            node_id_user = node_ids['user'].index(user_id)
            if len(user_timeline_tweets) > 0:
                for user_timeline_tweet_data in user_timeline_tweets:
                    if user_timeline_tweet_data['id'] not in node_ids['tweet']:
                        if start_time <= datetime.strptime(user_timeline_tweet_data['created_at'], '%a %b %d %H:%M:%S +0000 %Y') < end_time:
                            graph['tweet'].x[0].append(get_tweet_features(user_timeline_tweet_data)[0])
                            graph['tweet'].x[1].append(get_tweet_features(user_timeline_tweet_data)[1:])
                        else:
                            graph['tweet'].x[0].append('')
                            graph['tweet'].x[1].append([0]*2)
                        node_ids['tweet'].append(user_timeline_tweet_data['id'])
                    node_id_tweet = node_ids['tweet'].index(user_timeline_tweet_data['id'])
                    graph['user', 'posts', 'tweet'].edge_index[0] += [node_id_user]
                    graph['user', 'posts', 'tweet'].edge_index[1] += [node_id_tweet]
    graph['article'].x = torch.tensor(text_embeddings(graph['article'].x), dtype=torch.float32)
    graph['article'].y = torch.tensor(graph['article'].y, dtype=torch.long)
    if include_tweets:
        if include_text and np.asarray(graph['tweet'].x[1]).shape[0] > 0:
            graph['tweet'].x = torch.tensor(np.concatenate((text_embeddings(graph['tweet'].x[0]), np.asarray(graph['tweet'].x[1])), axis=1), dtype=torch.float32)
        else:
            graph['tweet'].x = torch.tensor(graph['tweet'].x[1], dtype=torch.float32)
        graph['tweet', 'cites', 'article'].edge_index = torch.tensor(graph['tweet', 'cites', 'article'].edge_index, dtype=torch.long)
    if include_users:
        if include_text and np.asarray(graph['user'].x[1]).shape[0] > 0:
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




def create_evolving_heterogeneous_snapshot(news_id_dict, tweet_id_list, start_time, end_time,
                                           dataset='politifact', include_tweets=True, include_users=True,
                                           include_user_timeline_tweets=True, include_retweets=True,
                                           include_user_followers=True, include_user_following=True,
                                           add_new_users=False, to_undirected=True, include_text=False):
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

    for subset, news_id in news_id_dict.items():
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
                tweets_path, tweet_list = get_news_tweet_ids(news_id=news_id, dataset=dataset, subset=subset)
                for tweet in tweet_list:
                    tweet_data = open_tweet_json(tweets_path, tweet)
                    if tweet in tweet_id_list:
                        if tweet_data['id'] not in node_ids['tweet']:
                            graph['tweet'].x[0].append(get_tweet_features(tweet_data)[0])
                            graph['tweet'].x[1].append(get_tweet_features(tweet_data)[1:])
                            node_ids['tweet'].append(tweet_data['id'])
                            node_id_news = node_ids['article'].index(news_id)
                            node_id_tweet = node_ids['tweet'].index(tweet_data['id'])
                            graph['tweet', 'cites', 'article'].edge_index[0] += [node_id_tweet]
                            graph['tweet', 'cites', 'article'].edge_index[1] += [node_id_news]
                        else:
                            node_id_news = node_ids['article'].index(news_id)
                            node_id_tweet = node_ids['tweet'].index(tweet_data['id'])
                            graph['tweet', 'cites', 'article'].edge_index[0] += [node_id_tweet]
                            graph['tweet', 'cites', 'article'].edge_index[1] += [node_id_news]

                    if include_users:
                        user_information = get_user_information(tweet_data['user']['id'])
                        if user_information:
                            if tweet in tweet_id_list:
                                if user_information['id'] not in node_ids['user']:
                                    graph['user'].x[0].append(get_user_features(user_information)[0])
                                    graph['user'].x[1].append(get_user_features(user_information)[1:])
                                    node_ids['user'].append(user_information['id'])
                                    node_id_user = node_ids['user'].index(user_information['id'])
                                    node_id_tweet = node_ids['tweet'].index(tweet_data['id'])
                                    graph['user', 'posts', 'tweet'].edge_index[0] += [node_id_user]
                                    graph['user', 'posts', 'tweet'].edge_index[1] += [node_id_tweet]
                                else:
                                    node_id_user = node_ids['user'].index(user_information['id'])
                                    node_id_tweet = node_ids['tweet'].index(tweet_data['id'])
                                    graph['user', 'posts', 'tweet'].edge_index[0] += [node_id_user]
                                    graph['user', 'posts', 'tweet'].edge_index[1] += [node_id_tweet]

                    if include_retweets:
                        retweets_path, retweet_ids = get_retweet_ids(news_id=news_id, dataset=dataset, subset=subset)
                        if tweet in tweet_id_list and tweet in retweet_ids:
                            retweets_data = open_retweet_json(retweets_path, tweet)
                            for retweet in retweets_data:
                                if retweet['id'] not in node_ids['tweet']:
                                    if start_time <= datetime.strptime(retweet['created_at'],
                                                                       '%a %b %d %H:%M:%S +0000 %Y') < end_time:
                                        # retweets node features
                                        graph['tweet'].x[0].append(get_tweet_features(retweet)[0])
                                        graph['tweet'].x[1].append(get_tweet_features(retweet)[1:])
                                        node_ids['tweet'].append(retweet['id'])
                                        node_id_retweet = node_ids['tweet'].index(retweet['id'])
                                        node_id_tweet = node_ids['tweet'].index(tweet_data['id'])
                                        graph['tweet', 'retweets', 'tweet'].edge_index[0] += [node_id_retweet]
                                        graph['tweet', 'retweets', 'tweet'].edge_index[1] += [node_id_tweet]
                                else:
                                    if start_time <= datetime.strptime(retweet['created_at'],
                                                                       '%a %b %d %H:%M:%S +0000 %Y') < end_time:
                                        node_id_retweet = node_ids['tweet'].index(retweet['id'])
                                        node_id_tweet = node_ids['tweet'].index(tweet_data['id'])
                                        graph['tweet', 'retweets', 'tweet'].edge_index[0] += [node_id_retweet]
                                        graph['tweet', 'retweets', 'tweet'].edge_index[1] += [node_id_tweet]

                                if include_users:
                                    user_information = get_user_information(retweet['user']['id'])
                                    if user_information:
                                        if tweet in tweet_id_list and retweet['id'] in node_ids['tweet']:
                                            if user_information['id'] not in node_ids['user']:
                                                graph['user'].x[0].append(get_user_features(user_information)[0])
                                                graph['user'].x[1].append(get_user_features(user_information)[1:])
                                                node_ids['user'].append(user_information['id'])
                                                node_id_retweet = node_ids['tweet'].index(retweet['id'])
                                                node_id_user = node_ids['user'].index(user_information['id'])
                                                graph['user', 'posts', 'tweet'].edge_index[0] += [node_id_user]
                                                graph['user', 'posts', 'tweet'].edge_index[1] += [node_id_retweet]
                                            else:
                                                node_id_retweet = node_ids['tweet'].index(retweet['id'])
                                                node_id_user = node_ids['user'].index(user_information['id'])
                                                graph['user', 'posts', 'tweet'].edge_index[0] += [node_id_user]
                                                graph['user', 'posts', 'tweet'].edge_index[1] += [node_id_retweet]
        else:
            print(f"[WARNING] excluding sample with id {news_id} no news or tweets available")
            graph['article'].x = torch.tensor(graph['article'].x, dtype=torch.float32)
            return graph

    graph['article'].x = torch.tensor(text_embeddings(graph['article'].x), dtype=torch.float32)
    graph['article'].y = torch.tensor(graph['article'].y, dtype=torch.long)
    if include_tweets:
        if include_text and np.asarray(graph['tweet'].x[1]).shape[0] > 0:
            graph['tweet'].x = torch.tensor(
                np.concatenate((text_embeddings(graph['tweet'].x[0]), np.asarray(graph['tweet'].x[1])), axis=1),
                dtype=torch.float32)
        else:
            graph['tweet'].x = torch.tensor(graph['tweet'].x[1], dtype=torch.float32)
        graph['tweet', 'cites', 'article'].edge_index = torch.tensor(graph['tweet', 'cites', 'article'].edge_index,
                                                                     dtype=torch.long)
    if include_users:
        if include_text and np.asarray(graph['user'].x[1]).shape[0] > 0:
            graph['user'].x = torch.tensor(
                np.concatenate((text_embeddings(graph['user'].x[0]), np.asarray(graph['user'].x[1])), axis=1),
                dtype=torch.float32)
        else:
            graph['user'].x = torch.tensor(graph['user'].x[1], dtype=torch.float32)
        graph['user', 'posts', 'tweet'].edge_index = torch.tensor(graph['user', 'posts', 'tweet'].edge_index,
                                                                  dtype=torch.long)
    if include_user_followers or include_user_following:
        graph['user', 'follows', 'user'].edge_index = torch.tensor(graph['user', 'follows', 'user'].edge_index,
                                                                   dtype=torch.long)
    if include_retweets:
        graph['tweet', 'retweets', 'tweet'].edge_index = torch.tensor(graph['tweet', 'retweets', 'tweet'].edge_index,
                                                                      dtype=torch.long)
    graph = graph.coalesce()
    if to_undirected:
        graph = T.ToUndirected(merge=False)(graph)
    return graph
