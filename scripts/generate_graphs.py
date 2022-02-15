from tqdm import tqdm
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from data_preprocessing.graph_structure import *

# generates graph for GossipCop dataset

ids_true, ids_fake = get_news_ids(dataset='gossipcop')

print("Starting with real news...")
for id in tqdm(list(ids_true)):
    graph = create_heterogeneous_graph({'real': [id]}, dataset='gossipcop', include_user_followers=False, include_user_following=False, include_retweets=True, include_user_timeline_tweets=True, to_undirected=True, include_text=True,
                                      include_users=True)
    if graph['article'].x.size()[0] > 0:
        graph_to_pickle(graph, id)


print("Starting with fake news...")
for id in tqdm(list(ids_fake)):
    graph = create_heterogeneous_graph({'fake': [id]}, dataset='gossipcop', include_user_followers=False, include_user_following=False, include_retweets=True, include_user_timeline_tweets=True, to_undirected=True, include_text=True,
                                      include_users=True)
    if graph['article'].x.size()[0] > 0:
        graph_to_pickle(graph, id)
