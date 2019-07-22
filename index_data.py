import es_connector as ES_Connector
from elasticsearch.helpers import bulk
import argparse
import json


def gen_index_data(num_groups, num_clusters):

	 # Load index json file
	 file_path = "./encode_results/encode_" + str(num_groups) \
	 			+ "groups_" + str(num_clusters) + "clusters/" \
	 			+ "encode_" + str(num_groups) + "groups_" \
	 			+ str(num_clusters) + "clusters" + ".json"
	 
	 # Open file
	 with open(file_path) as f:
	 	json_index_file = json.load(f)

	 for idx, document in enumerate(json_index_file):
	 	# Convert dict to JSON-based string
	 	document = json.dumps(document)
	 	
	 	# generate each document for indexing
	 	yield {
	 		"op_type": "index",
	 		"_index": "face_off",
	 		"_type": "_doc",
	 		"_id": str(idx),
	 		"_source": document
	 	}


def main():

	host = "localhost"
	port = 9200
	es = ES_Connector.connect(host, port)

	num_groups = 16
	num_clusters = 20
	bulk(es, gen_index_data(num_groups, num_clusters))


if __name__ == '__main__':
	main()












