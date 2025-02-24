
import os, time
from mistralai import Mistral
from chromadb import Documents, EmbeddingFunction, Embeddings
# import chromadb.utils.embedding_functions as embedding_functions


if not os.environ.get("MISTRAL_API_KEY"):
    print('ERR. `MISTRAL_API_KEY` env var is not set.')


class MistralEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        e = Embed()
        return [e.text(doc) for doc in input]


class Embed:
    def __init__(self):
        api_key = os.environ.get("MISTRAL_API_KEY")

        self.client = Mistral(api_key=api_key)

    def text(self, text: str) -> list[float]:
        ts = time.time()
        embeddings_batch_response = self.client.embeddings.create(
            model='mistral-embed',
            inputs=[text],
        )
        ts = time.time() - ts
        print(f'Took to create an embedding for text: {ts}s') 

        return embeddings_batch_response.data[0].embedding


