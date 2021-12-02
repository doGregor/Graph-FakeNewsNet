from data_preprocessing.load_data import *
from data_preprocessing.graph_structure import *
from data_preprocessing.visualization import *
from machine_learning.gnn_models import *
from machine_learning.gnn_training import *
from data_preprocessing.feature_extraction import *
from tqdm import tqdm


true, fake = get_news_ids(dataset='politifact')

print("Starting with real news...")
for id in tqdm(list(true)):
    graph = create_heterogeneous_graph({'real': [id]})
    if graph['article'].x.size()[0] > 0:
        graph_to_pickle(graph, true[0])

print("Starting with fake news...")
for id in tqdm(list(fake)):
    graph = create_heterogeneous_graph({'fake': [id]})
    if graph['article'].x.size()[0] > 0:
        graph_to_pickle(graph, true[0])
