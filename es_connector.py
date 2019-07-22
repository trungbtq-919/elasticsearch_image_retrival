from elasticsearch import Elasticsearch

def connect(host: str, port: int):
	es = Elasticsearch(host + ':' + str(port))

	return es