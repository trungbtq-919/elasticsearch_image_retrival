from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import argparse
import json
# import json_generator as JSON_Generator


class ESIndexer(object):

	def __init__(self, num_groups, num_clusters, server_url):
		self.server_url = server_url
		self.num_groups = num_groups
		self.num_clusters = num_clusters
		self.file_path =  "./encode_results/encode_" + str(num_groups) \
				+ "groups_" + str(num_clusters) + "clusters/" \
				+ "encode_" + str(num_groups) + "groups_" \
				+ str(num_clusters) + "clusters" + ".json"

	def gen_index_data(self):
		# Open file
		with open(self.file_path) as f:
			# print(self.file_path)
			json_index_file = json.load(f)

		for idx, document in enumerate(json_index_file):
			# Convert dict to JSON-based string
			document = json.dumps(document)

			# generate each document for indexing
			yield {
				"op_type": "index",
				"_index": "face_off_"+str(self.num_groups)+'groups_'+str(self.num_clusters)+'clusters',
				"_type": "_doc",
				"_id": str(idx),
				"_source": document
			}

	def index(self):

		es = Elasticsearch(self.server_url)
		bulk(es, self.gen_index_data())
		print("Index data successfully with {} groups and {} clusters".format(self.num_groups, self.num_clusters))
		print("*************")