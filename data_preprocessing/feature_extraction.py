from flair.embeddings import TransformerDocumentEmbeddings, WordEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence
import numpy as np


def text_embeddings(text_array, model='bert-base-uncased'):
    embedding = TransformerDocumentEmbeddings(model)
    embedding_array = []
    for sentence in text_array:
        sent = Sentence(sentence)
        embedding.embed(sent)
        embedding_array.append(sent.embedding.cpu().detach().numpy())
    return embedding_array
