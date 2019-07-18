"""
This file is used for testing search with Elasticsearch.
Author: Gia Huy
"""

from elasticsearch import Elasticsearch
import numpy as np
import csv
import json

def main():
	# Elasticsearch client.
	es = Elasticsearch("192.168.19.71:9200")

	# For testing
	# Query vector
	training_embedding_vectors = np.load("train_embs.npy")
	query_vector = training_embedding_vectors[0] # 1-st vector
	query_string_tokens = list()
	
	path = "./encode_results" + "/" \
			+ "encode_16groups_16clusters" + "/" \
			+ "encoded_string_16groups_16clusters" + ".csv"
	with open(path) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=",")
		for idx, line in enumerate(csv_reader):
			line_number = idx + 1
			if line_number == 1:
				query_string_tokens.append(line)
				break

	#print(type(query_string_tokens[0][0])) # For debug
	string_tokens_body = list()
	num_subvectors = 16
	for i in range(num_subvectors):
		sub_field = {
			"filter": {
				"term": {
					"string_token": query_string_tokens[0][i]
				}
			},
			"weight": 1
		}

		string_tokens_body.append(sub_field)

	s = 20
	r = 20
	request_body_1 = {
		"size": s,
		"query": {
			"function_score": {
				"functions": string_tokens_body,
				"score_mode": "sum",
				"boost_mode": "replace"
			}
		},
		"rescore": {
			"window_size": r,
			"query": {
				"rescore_query": {
					"function_score": {
						"script_score": {
							"script": {
								"lang": "painless",
								"source": """
									double sum = 0.0 ;
									for (int index = 0; index < doc['embedding_vector'].length; index++) {
										sum += Math.pow(params.query_vector[index] - doc['embedding_vector'][index], 2);
									}
									return(Math.sqrt(sum));
								""",
								"params": {
									"query_vector": query_vector.tolist()	
								}
							}
						},
						"boost_mode": "replace"
					}
				},
				"query_weight": 0,
				"rescore_query_weight": 1
			}
		}
	}

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
	print(json.dumps(res, indent=2)) # For debug

if __name__ == "__main__":
	main()
