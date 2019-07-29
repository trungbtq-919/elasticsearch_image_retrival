from elasticsearch import Elasticsearch
from data_encoder import DataEncoder
import numpy as np
from scipy import spatial
import csv
import json


class Searcher(object):

	def __init__(self, threshold, num_groups, num_clusters, query_vector, server_url, index_name):
		self.threshold = threshold
		self.es = Elasticsearch(server_url)
		self.index_name = index_name
		self.num_groups = num_groups
		self.num_clusters = num_clusters
		self.query_vector = query_vector
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
			"size": 30,
			"query": {
				"function_score": {
					"functions": string_tokens_body,
					"score_mode": "sum",
					"boost_mode": "replace"
				}
			}
		}

		return request_body

	def get_result_from_es(self):

		query_string_tokens = self.get_string_tokens()
		request_body = self.get_query_request_body(query_string_tokens)
		res = self.es.search(index=self.index_name, body=request_body)

		# Print Results in console.
		results_from_es = []
		for result in res['hits']['hits']:
			results_from_es.append(result['_source'])

		return results_from_es

	def re_rank(self):

		results_from_es = self.get_result_from_es()
		result_dist = []
		final_results = []
		for result_from_es in results_from_es:
			result_actual_vector = np.asarray(result_from_es['image_actual_vector'])
			# result_dist.append(np.linalg.norm(self.query_vector-result_actual_vector))
			# result_dist.append(spatial.distance.cosine(self.query_vector, result_actual_vector))
			dist = np.linalg.norm(self.query_vector-result_actual_vector)
			# dist = spatial.distance.cosine(self.query_vector, result_actual_vector)
			if dist <= self.threshold:
				final_results.append(result_from_es)


		# sorted_index = np.argsort(result_dist)
		# for i in range(self.num_returns):
		# 	index = np.where(sorted_index==i)[0][0]
		# 	final_results.append(results_from_es[index])
		return final_results

	def search(self):
		return self.re_rank()

def main():

	training_embedding_vectors = np.load("train_embs.npy")
	query_vector = training_embedding_vectors[3000]
	searcher = Searcher(10, 16, 16, query_vector)
	searcher.re_rank()


if __name__ == "__main__":
	main()
