from elasticsearch import Elasticsearch
from data_encoder import DataEncoder as Encoder
import numpy as np
import argparse
import csv
import json


def search(embedding_vector, ES_client):
	
	########### ENCODING ##########
	encoder = Encoder()
	string_tokens = encoder.encode(embedding_vector)
	
	# # Just load created string tokens.
	# file_path = "./encode_results/encode_" + str(args.num_groups) \
	#  			+ "groups_" + str(args.num_clusters) + "clusters/" \
	#  			+ "encoded_string_" + str(args.num_groups) + "groups_" \
	#  			+ str(args.num_clusters) + "clusters" + ".csv"
	# with open(file_path) as csv_file:
	# 	csv_reader = csv.reader(csv_file, delimiter=",")
	# 	for idx, line in enumerate(csv_reader):
	# 		line_number = idx + 1
	# 		if line_number == 7:
	# 			for sub_token in line:
	# 				query_string_tokens.append(sub_token)
	# 			break

	#print(type(query_string_tokens[0][0])) # For debug
	
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

		string_tokens_chunks.append(sub_field)

	# RETRIEVE ONLY
	request_body = {
		"size": 5,
		"query": {
			"function_score": {
				"functions": string_tokens_chunks,
				"score_mode": "sum",
				"boost_mode": "replace"
			}
		}
	}


	# print(json.dumps(request_body_2, indent=2)) # For debug
	res = ES_client.search(index="face_off", body=request_body)
	
	# Print Results in console.
	print(json.dumps(res, indent=2)) # For debug	


def rerank(vectors):
	pass


def get_distance(vector1, vector2, metric='Euclidean'):
	pass


def main():

	# Elasticsearch client.
	host = "192.168.19.71"
	port = "9200"
	es = Elasticsearch(host + ":" + port)

	# Query vector for testing purpose.
	model_folder_path = "hyperparams_training_data/"
	embedding_vectors = np.load(model_folder_path + "train_embs.npy")

	# Get i-th Embedding vector for searching.
	i = 0
	query_vector = np.expand_dims(embedding_vectors[i], axis=0)

	# Search
	search(query_vector, es)
	

if __name__ == "__main__":
	main()
