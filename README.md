# Heterogeneous Graphs for Fake News Detection

We evaluate how heterogeneous graphs constructed around news articles can
be used to detect fake stories. The contextual information describes 
social context and is modelled in network structure. In detail we use
 - news articles
 - user postings (tweets)
 - user repostings (retweets)
 - user accounts
 - user timeline-posts

as node types in our graphs and reformulate the problem as a 
graph classification task. We use the Politifact and Gossipcop 
datasets from FakeNewsNet (https://github.com/KaiDMML/FakeNewsNet).

## Project Structure:

### `data_preprocessing`

Python files to load and preprocess data (place a folder named `data` in the project's 
root directory that has two subfolders with the same structure as FakeNewsNet's 
`dataset` and `fakenewsnet_dataset` folders)

- `feature_extraction.py`: getting node related features like retweet count and generating transformer-based text embeddings
- `graph_structure.py`: functions to generate graphs from data. For an example see `scripts/generate_graphs.py`
- `load_data.py`: helper functions to load data from `data` folder during graph construction
- `text_summarization.py`: generating extractive and abstractive summaries from text (not used yet)
- `visualization.py`: function to visualize homogeneous graphs

### `machine_learning`
 
Python files that are related to graph machine learning

- `gnn_models.py`: GNNs used for experiments: SAGE, GAT, HGT. Architecture is currently adapted to graphs that feature all types of information (important for mean pooling node types)
- `gnn_training.py`: training and evaluation of models

### `scripts`

- `generate_graphs.py`: example script how to generate graphs. Parameters can be set to specify which node types should be considered
- `run_experiment.py`: example script that shows how the generated graphs can be used to run graph classification experiments
