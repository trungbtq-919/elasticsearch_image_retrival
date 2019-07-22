import es_connector as ES_Connector
from data_encoder import DataEncoder as Encoder
import numpy as np
import argparse
import csv
import json
import math
import logging
import sys


def search(query_vector, ES_client):
	""" Query vectors in database for similarity with `query_vector`

	* Note: Now, we just support for querying only one face at time.
	"""

	########### ENCODE ##########
	encoder = Encoder()
	string_tokens = encoder.encode(query_vector)

	string_tokens_chunks = list()
	for i in range(encoder.num_groups):
		sub_field = {
			"filter": {
				"term": {
					"string_token": string_tokens[i]
				}
			},
			"weight": 1
		}

		string_tokens_chunks.append(sub_field);

	# RETRIEVE ONLY
	s = 20
	request_body = {
		"size": s,
		"query": {
			"function_score": {
				"functions": string_tokens_chunks,
				"score_mode": "sum",
				"boost_mode": "replace"
			}
		}
	}

	############ QUERY #############
	res = ES_client.search(index="face_off", body=request_body)
	
	# Print some results in console for debugging.
	logging.debug(json.dumps(res, indent=2))

	############ RE-RANK ############
	vectors = []
	for i in range(s):
		embed_vector = res['hits']['hits'][i]['_source']['embedding_vector']
		id_ = res['hits']['hits'][i]['_id']

		# Convert to numpy array for reranking.
		embed_vector = np.array(embed_vector).reshape((1, -1))
		vectors.append(dict({
				'vector': embed_vector,
				'id': id_,
				'dist': 0
			}))
	
	rerank(vectors, anchor_vector=query_vector)
	top_id = [vector['id'] for vector in vectors]

	ret = []
	objs = res['hits']['hits']
	for id_ in top_id:
		for obj in objs:
			if obj['_id'] == id_:
				ret.append(obj)
				break

	return ret


def rerank(vectors, anchor_vector):
	s = len(vectors)
	for i in range(s):
		# Compare distance between two vector
		# using *Euclided distance* by default.
		dist = get_distance(vectors[i]['vector'], anchor_vector)
		vectors[i]['dist'] = dist
	
	# Ascending sort.
	vectors.sort(key=lambda x: x['dist'])


def get_distance(vector1, vector2, metric='Euclidean'):
	if metric == 'Euclidean':
		return math.sqrt(np.sum((vector2 - vector1)**2))


def logger_config(level):

	es_logger = logging.getLogger('elasticsearch')
	urllib_logger = logging.getLogger('urllib3')
	
	es_logger.setLevel(logging.WARNING)
	urllib_logger.setLevel(logging.WARNING)

	logging.basicConfig(level=level)


def main():

	logger_config(level=logging.WARNING)

	# Elasticsearch client.
	host = "localhost"
	port = 9200
	es = ES_Connector.connect(host, port)

	# Query vector for testing purpose.
	model_folder_path = "hyperparams_training_data/"
	embedding_vectors = np.load(model_folder_path + "train_embs.npy")

	# Get i-th Embedding vector for searching.
	i = 1
	query_vector = np.expand_dims(embedding_vectors[i], axis=0)

	# Search
	objs = search(query_vector, es)
	
	top_k = 5
	for i in range(top_k):
		print(json.dumps(objs[i], indent=2))

if __name__ == "__main__":
	main()













