import os
import time

from mistralai import Mistral

if not os.environ.get("MISTRAL_API_KEY"):
    print("ERR. `MISTRAL_API_KEY` env var is not set.")


class Embed:
    def __init__(self):
        api_key = os.environ.get("MISTRAL_API_KEY")

        self.client = Mistral(api_key=api_key)

    def text(self, text: str, opts: dict | None = None) -> list[float]:
        items = self.texts(texts=[text], opts=opts)
        return items[0]

    def texts(
        self, texts: list[str], opts: dict | None = None
    ) -> list[list[float]]:
        ts = time.time()
        embeddings_batch_response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=texts,
        )
        ts = time.time() - ts
        print(f"Took to create an embedding for text: {ts}s")

        return list(map(lambda x: x.embedding, embeddings_batch_response.data))
