"""
This file is used for testing search with Elasticsearch.
Author: Gia Huy
"""

from elasticsearch import Elasticsearch
from data_encoder import DataEncoder
import numpy as np
import csv
import json


class Searcher(object):

	def __init__(self, num_returns, num_groups, num_clusters, query_vector):
		self.num_returns = num_returns
		self.es = Elasticsearch("localhost:9200")
		self.num_groups = num_groups
		self.num_clusters = num_clusters
		self.data_encoder = DataEncoder(num_groups, num_clusters, 1000, query_vector.reshape(1, query_vector.shape[0]))

	def get_string_tokens(self):
		kmeans, query_string_tokens_list = self.data_encoder.encode_string_tokens()
		return query_string_tokens_list[0]

	def get_query_request_body(self, query_string_tokens):
		string_tokens_body = list()
		for i in range(self.num_groups):
			sub_field = {
				"filter": {
					"term": {
						"image_encoded_tokens": query_string_tokens[i]
					}
				},
				"weight": 1
			}

			string_tokens_body.append(sub_field)


		# RETRIEVE ONLY
		request_body = {
			"size": self.num_returns,
			"query": {
				"function_score": {
					"functions": string_tokens_body,
					"score_mode": "sum",
					"boost_mode": "replace"
				}
			}
		}

		return request_body

	def search(self, request_body):

		res = es.search(index="face_off", body=request_body)

		# Print Results in console.
		for result in res['hits']['hits']:
			print(result['_source']['id'])

def main():
	# Elasticsearch client.
	es = Elasticsearch("localhost:9200")

	# For testing
	# Query vector
	training_embedding_vectors = np.load("train_embs.npy")
	query_vector = training_embedding_vectors[6] # 1-st vector

	data_encoder = DataEncoder(16, 16, 1000, query_vector.reshape(1, query_vector.shape[0]))
	kmeans, query_string_tokens_list = data_encoder.encode_string_tokens()
	query_string_tokens = query_string_tokens_list[0]
	print(query_string_tokens)

	# query_string_tokens = list()
	#
	# path = "./encode_results" + "/" \
	# 		+ "encode_16groups_16clusters" + "/" \
	# 		+ "encoded_string_16groups_16clusters" + ".csv"
	#
	# with open(path) as csv_file:
	# 	csv_reader = csv.reader(csv_file, delimiter=",")
	# 	for idx, line in enumerate(csv_reader):
	# 		line_number = idx + 1
	# 		if line_number == 7:
	# 			print(line)
	# 			query_string_tokens = line
	# 			break

	#print(type(query_string_tokens[0][0])) # For debug
	string_tokens_body = list()
	num_subvectors = 16
	for i in range(num_subvectors):
		sub_field = {
			"filter": {
				"term": {
					"image_encoded_tokens": query_string_tokens[i]
				}
			},
			"weight": 1
		}

		string_tokens_body.append(sub_field)

	s = 20

	# RETRIEVE ONLY
	request_body_2 = {
		"size": s,
		"query": {
			"function_score": {
				"functions": string_tokens_body,
				"score_mode": "sum",
				"boost_mode": "replace"
			}
		}
	}

	# print(json.dumps(request_body_2, indent=2)) # For debug
	res = es.search(index="face_off", body=request_body_2)
	
	# Print Results in console.
	for result in res['hits']['hits']:
		print(result['_source']['id'])

if __name__ == "__main__":
	main()
