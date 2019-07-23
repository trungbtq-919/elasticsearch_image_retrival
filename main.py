from index_data import ESIndexer
from search import Searcher
from elasticsearch import Elasticsearch
import numpy as np


def main():
    num_groups = 16
    num_clusters = 32
    server_url = 'localhost:9200'
    num_returns = 10
    training_embedding_vectors = np.load("train_embs.npy")

    es = Elasticsearch(server_url)
    if not es.indices.exists('face_off_'+str(num_groups)+'groups_'+str(num_clusters)+'clusters'):
        indexer = ESIndexer(num_groups, num_clusters, server_url)
        indexer.index()
    else:
        index_name = 'face_off_'+str(num_groups)+'groups_'+str(num_clusters)+'clusters'
        query_vector = training_embedding_vectors[1]
        searcher = Searcher(num_returns, num_groups, num_clusters, query_vector, server_url, index_name)
        searcher.search()


if __name__ == '__main__':
    main()