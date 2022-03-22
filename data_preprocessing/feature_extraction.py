import os.path
import json
from flair.embeddings import TransformerDocumentEmbeddings, WordEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence
import numpy as np


EMBEDDING = TransformerDocumentEmbeddings('bert-base-cased')


def text_embeddings(text_array):
    embedding_array = []
    for sentence in text_array:
        if sentence == '' or sentence == ' ':
            embedding_array.append(np.zeros(768))
        else:
            sent = Sentence(sentence)
            EMBEDDING.embed(sent)
            embedding_array.append(sent.embedding.cpu().detach().numpy())
    return np.asarray(embedding_array)


def get_news_features(news_data):
    return news_data['title'] + '. ' + news_data['text']


def get_summaries(news_id, dataset='politifact', subset='fake', *args):
    file_path = '../data/fakenewsnet_dataset/' + dataset + '/' + subset + '/' + news_id + '/summary.json'
    output = ''
    if os.path.isfile(file_path):
        with open(file_path) as json_file:
            data = json.load(json_file)
        for summary_type in args:
            output += data[summary_type] + ' '
        return output.strip()
    else:
        return output


def get_tweet_features(tweet_data):
     return [tweet_data['text'], tweet_data['retweet_count'], tweet_data['favorite_count']]


def get_user_features(user_data):
    return [user_data['description'], user_data['followers_count'],  user_data['friends_count'], user_data['favourites_count'],
            user_data['statuses_count']]



if __name__ == '__main__':

    test_array = ['hello', 'test', 'a string', '']
    embedded_array = text_embeddings(test_array)
    print(embedded_array.shape)
    print(embedded_array[-1].shape)
