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


# Citation

The paper based on this idea was accepted at ECIR 2023. If you use parts of our code or adopt our approach we kindly ask you to cite our work as follows:
```
@inproceedings{10.1007/978-3-031-28238-6_29,
author = {Donabauer, Gregor and Kruschwitz, Udo},
title = {Exploring Fake News Detection with Heterogeneous Social Media Context Graphs},
year = {2023},
isbn = {978-3-031-28237-9},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-28238-6_29},
doi = {10.1007/978-3-031-28238-6_29},
booktitle = {Advances in Information Retrieval: 45th European Conference on Information Retrieval, ECIR 2023, Dublin, Ireland, April 2–6, 2023, Proceedings, Part II},
pages = {396–405},
numpages = {10},
location = {Dublin, Ireland}
}
```
