import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings


class Search:
    def new(self, collection_name):
        return _Search(collection_name)


from loader import load_config, load_module

_client = chromadb.HttpClient(
    host=load_config("search.yml")["hostname"],
    # host="chromadb",
    port=8000,
)

embed = load_module("embed.yml")


class _EmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        super().__init__()

    def name(self) -> str:
        return "SvenEmbeddingFunction"

    def get_config(self) -> dict:
        return {}

    def __call__(self, input: Documents) -> Embeddings:
        return [embed.text(doc) for doc in input]


class _Search:
    def __init__(self, collection_name):
        print(f"New Search for collection: {collection_name}")

        _client.heartbeat()

        emb_fn = _EmbeddingFunction()
        self.emb_fn = emb_fn

        collection = _client.get_or_create_collection(
            name=collection_name,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine", "hnsw:search_ef": 100},
        )
        self.collection = collection

        print(f"number of items in the collection: {collection.count()}")
        # print(f'the first 10 items in the collection:\nf{collection.peek()}')

    def add(
        self,
        text: str,
        id: str,
        metadata: dict[str, str],
        embedding: list[float] | None,
    ):
        # TODO: - do to add if already exists - check by ID
        # Save time not embedding

        if embedding is None:
            embedding = self.emb_fn([text])
        # embedding = [self.embed.text(text)]

        self.collection.add(
            documents=[text],
            embeddings=embedding,
            metadatas=[metadata],
            ids=[id],
        )

    def search(
        self,
        text: str,
        tags: list[str] | None = None,
        n: int = 30,
    ):
        return self.collection.query(
            query_texts=[text],
            n_results=n,
            # where={"metadata_field": "is_equal_to_this"},
            # where_document={"$contains":"search_string"}
        )

        #
        # Multi-Category/Tag Filters in chroma:
        # https://cookbook.chromadb.dev/strategies/multi-category-filters/
        #
        # "$or" example
        # https://cookbook.chromadb.dev/strategies/keyword-search/
        #
        # self.collection.query(
        #     query_texts=["technology"],
        #     where_document={
        #         "$or": [{"$contains": "technology"}, {"$contains": "freak"}]
        #     },
        # )

    def peek(self):
        return self.collection.peek()

    def delete(self, ids: list[str]):
        self.collection.delete(ids=ids)

    def get(self, ids: list[str]):
        return self.collection.get(ids=ids)
