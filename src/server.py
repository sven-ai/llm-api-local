import time
import uuid

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer

from config_models import *
from loader import load_module
from models_storage import *
from utils import *

access = load_module("access.yml")
neural_search = load_module("search.yml")

_db_models = DbModels()

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
collections = LimitedDict(max_size=100)


empty_search_results = {
    # "distances": [dist for dist in distances if dist < threshold],
    "documents": [],
    "ids": [],
    "metadatas": [],
}


def filter_search_results(contents, threshold: float = 0.3):
    # print(f'All (unfiltered) embeddings: {contents}')

    has_data = (
        ("distances" in contents)
        and (len(contents["distances"]) > 0)
        and (len(contents["distances"][0]) > 0)
    )
    if not has_data:
        print("Found 0 related docs.")
        return empty_search_results
        # return contents

    distances = contents["distances"][0]
    documents = contents["documents"][0]
    ids = contents["ids"][0]
    metadatas = contents["metadatas"][0]

    # print(f'All (unfiltered) distances: {distances}')

    def is_below_threshold(v):
        return v < threshold

    indices = [
        idx for idx, val in enumerate(distances) if is_below_threshold(val)
    ]

    return {
        # "distances": [dist for dist in distances if dist < threshold],
        "documents": [documents[idx] for idx in indices],
        "ids": [ids[idx] for idx in indices],
        "metadatas": [metadatas[idx] for idx in indices],
    }

    # documents = [v for v, dist in zip(documents, distances) if dist < threshold]
    # ids = [v for v, dist in zip(ids, distances) if dist < threshold]
    # metadatas = [v for v, dist in zip(metadatas, distances) if dist < threshold]

    # return {
    #     # "distances": [dist for dist in distances if dist < threshold],
    #     "documents": documents,
    #     "ids": ids,
    #     "metadatas": metadatas,
    # }


rerank = load_module("rerank.yml")


class SearchQuery(BaseModel):
    q: str
    n: int = 1


@app.post("/search")
def api_search(
    query: SearchQuery,
    token: str = Depends(oauth2_scheme),
):
    if not access.bearer_is_valid(token):
        raise HTTPException(
            status_code=403, detail="Access Denied: Invalid Bearer Token"
        )

    res = search(query.q, token)
    if "documents" in res and len(res["documents"]) > 0:
        return res["documents"][0]
    else:
        return "No related knowledge."


def search(q: str, collection: str):
    print(f"Searching in c: {collection} | q: {q}")

    neural_searcher = collections.get_or_insert(
        user_modelname_to_embedding_modelname(collection),
        lambda x: neural_search.new(x),
    )

    ts = time.time()
    res = neural_searcher.search(q)
    ts = time.time() - ts
    print(f"Took to search using q text: {ts}s")

    filtered = filter_search_results(res)
    n = len(filtered["documents"])
    # print(f'SEARCH: {filtered}')

    top_n = 10
    if n > top_n:
        ts = time.time()
        ranked = rerank.rank(q, filtered, top_n)
        ts = time.time() - ts
        print(f"Took to rerank {n} docs: {ts}s")
        # print(f'ranked res: {ranked}')

        return ranked
    else:
        print(f"Search found {n} related docs. Reranking skipped.")
        return filtered


class AddEmbedItem(BaseModel):
    model: str  # collection_name
    input: str
    metadata: dict[str, str]


@app.post("/v1/embeddings")
async def embed_add_v1(
    item: AddEmbedItem,
    id: str = None,
    token: str = Depends(oauth2_scheme),
):
    if not access.bearer_is_valid(token):
        raise HTTPException(
            status_code=403, detail="Access Denied: Invalid Bearer Token"
        )

    if len(item.input) == 0 or len(item.model) == 0:
        raise HTTPException(status_code=500, detail="Bad input")

    return {
        "object": "embeddings",
        "id": uuid.uuid4(),
        "model": item.model,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "data": embed.text(item.input),
    }


known_models = set()


@app.post("/v2/embeddings")
async def embed_add(
    item: AddEmbedItem,
    id: str = None,
    token: str = Depends(oauth2_scheme),
):
    if not access.bearer_is_valid(token):
        raise HTTPException(
            status_code=403, detail="Access Denied: Invalid Bearer Token"
        )

    if len(item.input) == 0 or len(item.model) == 0:
        raise HTTPException(status_code=500, detail="Bad input")

    model_item = ModelItem(
        id=item.model,
        name=item.metadata.get("name", f"{item.model} knowledge base"),
        description=item.metadata.get(
            "description", f"{item.model} generic about"
        ),
    )
    if item.model in known_models:
        #
        # TODO - do not update always
        # only when values did change
        #
        _db_models.update_metadata(model_item)
    else:
        known_models.add(item.model)

        print(f"Registering new model: {item.model}")
        _db_models.add_new_if_needed(model_item)

    # models = known_bearers[token]
    # if item.model not in models:
    #     raise HTTPException(status_code=403, detail="Access Denied: Invalid model")
    model = user_modelname_to_embedding_modelname(item.model)
    print(f"Adding knowledge to: {model}")

    neural_searcher = collections.get_or_insert(
        model,  # item.model,
        lambda x: neural_search.new(x),
    )

    if id is None:
        id = str(uuid.uuid4())

    # {"key": "value"}
    neural_searcher.add(item.input, id, item.metadata)
    return {"id": id}


@app.get("/v1/models")
def models_v1(
    token: str = Depends(oauth2_scheme),
):
    raise HTTPException(status_code=301, detail="USE /v2 not /v1")


@app.get("/v2/models")
def models(
    token: str = Depends(oauth2_scheme),
):
    if not access.bearer_is_valid(token):
        raise HTTPException(
            status_code=403, detail="Access Denied: Invalid Bearer Token"
        )

    models = list(_db_models.list_all())
    if len(models) > 0:
        # For every model - insert 3 `reasoning`-supported models:
        #
        rms = []
        for m in models:
            rms.append(
                ModelItem(
                    id=f"*{m.id}",
                    name=f"[reason-1] {m.name}",
                    description=f"{m.description} and Think up to 5K tokens.",
                )
            )
            rms.append(
                ModelItem(
                    id=f"**{m.id}",
                    name=f"[reason-2] {m.name}",
                    description=f"{m.description} and Think up to 15K tokens.",
                )
            )
            rms.append(
                ModelItem(
                    id=f"***{m.id}",
                    name=f"[reason-3] {m.name}",
                    description=f"{m.description} and Think up to 30K tokens.",
                )
            )
    models.extend(rms)

    return {
        "object": "list",
        "data": list(map(lambda x: model_config(x), models)),
    }


llm_provider = load_module("llmprovider.yml")
from llm import *
from loader import load_module

embed = load_module("embed.yml")


def fixedChatCompletionsPairItem(
    x: ChatCompletionsPairItem,
) -> ChatCompletionsPairItem:
    return (
        ChatCompletionsPairItem(role=x.role, content="...")
        if x.content == ""
        else x
    )


@app.post("/v1/chat/completions")
def completions_v1(
    item: ChatCompletionsItem,
    token: str = Depends(oauth2_scheme),
):
    if (
        item.model == "tmp"
        or item.model == "*tmp"
        or item.model == "**tmp"
        or item.model == "***tmp"
    ):
        return completions(item, token)
    else:
        return LLM().plaintext_content_response(
            item.model,
            "Use /v2 API endpoint, not /v1. https://api.svenai.com/v2",
        )


@app.post("/v2/chat/completions")
def completions(
    item: ChatCompletionsItem,
    token: str = Depends(oauth2_scheme),
):
    if not access.bearer_is_valid(token):
        raise HTTPException(
            status_code=403, detail="Access Denied: Invalid Bearer Token"
        )

    # print(f"completions input item: {item}")
    llm = LLM()

    messages = list(
        map(lambda x: fixedChatCompletionsPairItem(x), item.messages)
    )
    # # print(f'model: {model}')
    # print(f'fixed messages: {messages}')

    # remake with `fixed` messages - `fixed` means no message has empty content
    item.messages = messages

    if len(messages) == 0:
        return llm.plaintext_content_response(
            item.model, "Be nice and ❤️ Jessica."
        )
    q = messages[-1].content
    # print(f'q: {q}')

    embeddings = None
    if (
        item.model == "tmp"
        or item.model == "*tmp"
        or item.model == "**tmp"
        or item.model == "***tmp"
    ):
        # `tmp` blackhole model has no knowledge
        embeddings = empty_search_results
    else:
        embeddings = search(
            q=q,
            collection=item.model,
        )
    assert embeddings is not None

    # print(f"Forwarding request to LLM: {item}")
    return llm.forward_to_llm(llm_provider, item, embeddings)


dummy_html_200 = """
<!DOCTYPE html> <html lang="en">
    <head> <meta charset="UTF-8"> <meta name="viewport" content="width=device-width, initial-scale=1.0"> <title>Sven AI</title>
    </head>

    <body>
        Sven - own your knowledge data.
        <a href="https://svenai.com/own-your-knowledge-data/">Learn More</a>
    </body>
</html>
"""


@app.get("/")
async def dummy_head_html():
    return HTMLResponse(content=dummy_html_200, status_code=200)


@app.get("/v2")
async def dummy_api_html():
    return HTMLResponse(content=dummy_html_200, status_code=200)


#
# OLLAMA endpoints support
#


@app.head("/")
async def ollama_head(
    token: str = Depends(oauth2_scheme),
):
    if not access.bearer_is_valid(token):
        raise HTTPException(
            status_code=403, detail="Access Denied: Invalid Bearer Token"
        )

    return HTMLResponse(content="<html><body>OK</body></html>", status_code=200)


@app.get("/api/ps")
@app.get("/api/tags")
def ollama_models(
    token: str = Depends(oauth2_scheme),
):
    if not access.bearer_is_valid(token):
        raise HTTPException(
            status_code=403, detail="Access Denied: Invalid Bearer Token"
        )

    mm = models(token=token)

    # print(f'mm: {mm}')
    data = mm["data"]
    # print(f'mm.data: {data}')

    return {
        "models": list(
            map(
                lambda x: {
                    "name": x["id"],
                    "model": x["name"],
                    "modified_at": "2023-11-04T14:56:49.277302595-07:00",
                    "size": 7365960935,
                    "digest": x["id"],
                    "details": {
                        "format": "gguf",
                        "parent_model": "llama",
                    },
                },
                data,
            )
        )
    }


@app.post("/api/chat")
def ollama_chat(
    item: ChatCompletionsItem,
    token: str = Depends(oauth2_scheme),
):
    if not access.bearer_is_valid(token):
        raise HTTPException(
            status_code=403, detail="Access Denied: Invalid Bearer Token"
        )
    # print(f'ollama input item: {item}')

    cc = completions(
        item,
        token,
    )
    # print(type(cc))
    # print(f'cc: {cc}')

    content = None
    choices = cc["choices"]
    if choices is not None and len(choices) > 0:
        choice = choices[0]
        content = choice["message"]["content"]
    else:
        print("ollama cant get choices.")

    if content is None:
        content = "Ollama failed :("
    # else:
    #     print(f'parsed content: {content}')

    return {
        "model": "anton",
        # "created_at": "2023-08-04T08:52:19.385406455-07:00",
        "message": {
            "role": "assistant",
            "content": content,
            # "images": null
        },
        "done": True,
    }


if __name__ == "__main__":
    import sys

    import uvicorn

    # API server
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        uvicorn.run(
            # live code reload
            # https://github.com/encode/uvicorn/issues/687
            "server:app",
            reload=True,
            workers=1,
            host="0.0.0.0",
            port=12345,
        )
    else:
        # production env
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=12345,
        )
