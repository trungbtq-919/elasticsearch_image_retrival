"""
This file is used for testing search with Elasticsearch.
Author: Gia Huy

This program takes 2 argument from command line:
	1. arg1: num_groups (type: int)
	2. arg2: num_clusters (type: int)
"""

from elasticsearch import Elasticsearch
import numpy as np
import argparse
import csv
import json

def main():
	parser = argparse.ArgumentParser("Search documents.")
	parser.add_argument("--num_groups", type=int, required=True)
	parser.add_argument("--num_clusters", type=int, required=True)
	args = parser.parse_args()

	# Elasticsearch client.
	host = "192.168.19.71"
	port = "9200"
	es = Elasticsearch(host + ":" + port)

	# For testing
	# Query vector
	training_embedding_vectors = np.load("train_embs.npy")
	query_vector = training_embedding_vectors[6] # 1-st vector
	query_string_tokens = list()
	
	# Just load created string tokens.
	file_path = "./encode_results/encode_" + str(args.num_groups) \
	 			+ "groups_" + str(args.num_clusters) + "clusters/" \
	 			+ "encoded_string_" + str(args.num_groups) + "groups_" \
	 			+ str(args.num_clusters) + "clusters" + ".csv"
	with open(file_path) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=",")
		for idx, line in enumerate(csv_reader):
			line_number = idx + 1
			if line_number == 7:
				for sub_token in line:
					query_string_tokens.append(sub_token)
				break

	#print(type(query_string_tokens[0][0])) # For debug
	string_tokens_body = list()
	for i in range(args.num_groups):
		sub_field = {
			"filter": {
				"term": {
					"string_token": query_string_tokens[i]
				}
			},
			"weight": 1
		}

		string_tokens_body.append(sub_field)

	s = 6
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
			"window_size": r, # Get top-r results from query phase for rescoring with Eucliean distance.
			"query": {
				"rescore_query": {
					"function_score": {
						"script_score": {
							"script": {
								"lang": "painless",
								"source": """
									def sum = 0.0 ;
									for (def index = 0; index < params['_source']['embedding_vector'].length; index++) {
										sum += Math.pow(params.query_vector[index] - doc['embedding_vector'][index], 2);
									}
									return(Math.sqrt(sum));
								""",
								"params": {
									"query_vector": query_vector.tolist() # numpy array not working here.
								}
							}
						},
						"boost_mode": "replace"
					}
				},
				"query_weight": 0, # Remove scores from query phase.
				"rescore_query_weight": 1 # Just calculate scores according to *rescoring phase*.
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

	request_body_3 = {
		"size": s,
		"query": {
			"function_score": {
				"functions": string_tokens_body,
				"score_mode": "sum",
				"boost_mode": "replace"
			}
		},
		"rescore": {
			"window_size": r, # Get top-r results from query phase for rescoring with Eucliean distance.
			"query": {
				"rescore_query": {
					"function_score": {
						# "query": {
						# 	"function_score": {
						# 		"functions": string_tokens_body,
						# 		"score_mode": "sum",
						# 		"boost_mode": "replace"
						# 	}
						# },
						"script_score": {
							"script": {
								# "lang": "painless" ,
								"source": """
									def sum = 0.0;
									for (def i = 0; i < params['_source']['embedding_vector'].length; i++) {
										sum += Math.pow((params['_source']['embedding_vector'][i] - params.query_vector[i]), 2);
									}
									return (Math.sqrt(sum));
								""",
								"params": {
									"query_vector": query_vector.tolist() # numpy array not working here.
								}
								# "inline": "doc.label.value"
							}
						},
						# "boost_mode": "replace"
					}
				},
				"query_weight": 0, # Remove scores from query phase.
				"rescore_query_weight": 1 # Just calculate scores according to *rescoring phase*.
			}
		}
	}

	# print(json.dumps(request_body_2, indent=2)) # For debug
	res = es.search(index="face_off", body=request_body_2)
	
	# Print Results in console.
	print(json.dumps(res, indent=2)) # For debug

if __name__ == "__main__":
	main()


"""
doc['embedding_vector'].length
def sat_scores = [];
 def scores = ['AvgScrRead', 'AvgScrWrit', 'AvgScrMath'];
  for (int i = 0; i < scores.length; i++) {
  	sat_scores.add(doc[scores[i]].value)
  } 
def temp = 0;
 for (def score : sat_scores) {
 	temp += score;
 } 
 sat_scores.add(temp);
  return sat_scores;

"""