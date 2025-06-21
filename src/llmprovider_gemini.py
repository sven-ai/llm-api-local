import os

from llm import *
from utils import *

if not os.environ.get("GEMINI_API_KEY"):
    print("ERR. `GEMINI_API_KEY` env var is not set.")


class Llmprovider(LLMProviderBase):
    supportsStreaming = True

    def __init__(self):
        self.client = None

    def req(
        self,
        llm: LLM,
        item: ChatCompletionsItem,
        embeddings,
    ):
        # TODO: - impl
        return None
