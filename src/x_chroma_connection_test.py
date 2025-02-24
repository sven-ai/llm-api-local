import chromadb

_client = chromadb.HttpClient(host='chromadb', port=8000)
_client.heartbeat()