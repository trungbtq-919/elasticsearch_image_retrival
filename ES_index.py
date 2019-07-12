from elasticsearch import Elasticsearch
import requests, json, os

res = requests.get('http://localhost:9200')
print (res.content)
# connect to ES cluster
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

json_file_path = './encode_results/encode_16groups_16clusters/encode_16groups_16clusters.json'
f = open(json_file_path)
string_tokens_content = f.read()
es.index(index='string_tokens_16_16', doc_type='encode_16_16', body=json.loads(string_tokens_content))
