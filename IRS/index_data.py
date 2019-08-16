from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import argparse
import json
# import json_generator as JSON_Generator


class ESIndexer(object):

	def __init__(self, directory, num_groups, num_clusters, server_url, model_name=None):
		self.server_url = server_url
		self.num_groups = num_groups
		self.num_clusters = num_clusters
		self.file_path = "./" + directory + "/encode_" + str(num_groups) \
				+ "groups_" + str(num_clusters) + "clusters/" \
				+ "encode_" + str(num_groups) + "groups_" \
				+ str(num_clusters) + "clusters" + ".json"
		self.model_name = model_name

	def gen_index_data(self):
		# Open file
		with open(self.file_path) as f:
			# print(self.file_path)
			json_index_file = json.load(f)

		if self.model_name == None:
			index_name = "face_off_" + str(self.num_groups) + 'groups_' + str(self.num_clusters) + 'clusters'
		else:
			index_name = "face_off_"+str(self.num_groups)+'groups_'+str(self.num_clusters)+'clusters_' + self.model_name

		for idx, document in enumerate(json_index_file):
			# Convert dict to JSON-based string
			document = json.dumps(document)

			# generate each document for indexing
			yield {
				"op_type": "index",
				"_index": index_name,
				"_type": "_doc",
				"_id": str(idx),
				"_source": document
			}

	def index(self):

		es = Elasticsearch(self.server_url)
		bulk(es, self.gen_index_data())
		print("Index data successfully with {} groups and {} clusters".format(self.num_groups, self.num_clusters))
		print("*************")