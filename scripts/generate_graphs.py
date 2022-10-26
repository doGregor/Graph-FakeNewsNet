from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from data_preprocessing.graph_structure import *


# generates graph for GossipCop dataset

DATASET = 'gossipcop'

ids_true, ids_fake = get_news_ids(dataset=DATASET)

setups = ['tweets_only'] # 'all_data' 'no_retweets' 'no_timeline' 'tweets_users' 'tweets_only'

for s in setups:
    print(s)
    
    if s == 'all_data':
        include_retweets = True
        include_user_timeline_tweets = True
        include_users = True
        include_tweets = True
    if s == 'no_retweets':
        include_retweets = False
        include_user_timeline_tweets = True
        include_users = True
        include_tweets = True
    if s == 'no_timeline':
        include_retweets = True
        include_user_timeline_tweets = False
        include_users = True
        include_tweets = True
    if s == 'tweets_only':
        include_retweets = False
        include_user_timeline_tweets = False
        include_users = False
        include_tweets = True
    if s == 'tweets_users':
        include_retweets = False
        include_user_timeline_tweets = False
        include_users = True
        include_tweets = True
        
    def save_graph_to_pickle(graph, file_name):
        path = "static_graphs/" + DATASET + '/' + s + '/' + file_name + ".pickle"
        with open(path, 'wb') as handle:
            pickle.dump({'graph': graph}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Starting with real news...")
    for id in tqdm(list(ids_true)):
        try:
            graph = create_heterogeneous_graph({'real': [id]},
                                               dataset=DATASET,
                                               include_tweets=include_tweets,
                                               include_user_followers=False,
                                               include_user_following=False,
                                               include_retweets=include_retweets,
                                               include_user_timeline_tweets=include_user_timeline_tweets,
                                               to_undirected=True,
                                               include_text=True,
                                               include_users=include_users)
            if graph['article'].x.size()[0] > 0:
                save_graph_to_pickle(graph, id)
        except:
            print('ERROR', id)


    print("Starting with fake news...")
    for id in tqdm(list(ids_fake)):
        try:
            graph = create_heterogeneous_graph({'fake': [id]},
                                               dataset=DATASET,
                                               include_tweets=include_tweets,
                                               include_user_followers=False,
                                               include_user_following=False,
                                               include_retweets=include_retweets,
                                               include_user_timeline_tweets=include_user_timeline_tweets,
                                               to_undirected=True,
                                               include_text=True,
                                               include_users=include_users)
            if graph['article'].x.size()[0] > 0:
                save_graph_to_pickle(graph, id)
        except:
            print('ERROR', id)
