import es_connector as ES_Connector

def update(vector, str_tokens):
	host = "localhost"
	port = 9200
	es = ES_Connector.connect(host, port)

	