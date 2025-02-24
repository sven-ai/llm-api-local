import chromadb
# from embed import Embed
from embed import MistralEmbeddingFunction

_client = chromadb.HttpClient(host='dev-falk', port=8000)

class Search:
    def __init__(self, collection_name):
        print(f'New NeuralSearcher for collection: {collection_name}')
        
        _client.heartbeat()

        emb_fn = MistralEmbeddingFunction()
        self.emb_fn = emb_fn

        collection = _client.get_or_create_collection(
            name=collection_name, 
            embedding_function=emb_fn,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:search_ef": 100
            }
        )
        self.collection = collection

        print(f'number of items in the collection: {collection.count()}')
        # print(f'the first 10 items in the collection:\nf{collection.peek()}')


    def add(
        self, 
        text: str, id: str, 
        metadata: dict[str, str],
        ):
        embedding = self.emb_fn([text])
        # embedding = [self.embed.text(text)]

        self.collection.add(
            documents=[text],
            embeddings=embedding,
            metadatas=[metadata],
            ids=[id]
        )



    def search(
        self, 
        text: str,
        n: int = 30,
        ):
        return self.collection.query(
            query_texts=[text],
            n_results=n,
            # where={"metadata_field": "is_equal_to_this"},
            # where_document={"$contains":"search_string"}
        )
