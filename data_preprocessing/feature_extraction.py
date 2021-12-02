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
    return np.asarray(embedding_array)


def get_news_features(news_data):
    return news_data['text']


def get_tweet_features(tweet_data):
    # return [tweet_data['text'], tweet_data['retweet_count'], tweet_data['favorite_count']]
    return [tweet_data['retweet_count'], tweet_data['favorite_count']]


def get_user_features(user_data):
    #return [user_data['description'], user_data['followers_count'],  user_data['friends_count'], user_data['favourites_count'],
    #        user_data['statuses_count']]
    return [user_data['followers_count'], user_data['friends_count'], user_data['favourites_count'], user_data['statuses_count']]


if __name__ == '__main__':
    test_array = ['hello', 'test', 'an string']
    embedded_array = text_embeddings(test_array)
    print(len(embedded_array))
    print(embedded_array[0].shape)
