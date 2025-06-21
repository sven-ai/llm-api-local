import datetime
import json
import time
import uuid

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from config_models import *
from loader import load_module
from models_storage import *
from utils import *

rerank = load_module("rerank.yml")
access = load_module("access.yml")
neural_search = load_module("search.yml")
from mcp_shared import *

mcp_provider = load_module("mcp.yml")

_db_models = DbModels()

# mcp = FastMCP(
#     name="DevBrain - Developer's Knowledge MCP Server",
#     instructions="Provides tools for knowledge and context discovery. Call `devbrain_find_knowledge()` and pass a question to retrieve related information. Results may include hints, tips, guides or code snippets. DevBrain's provides up-to-date knowledge curated by real software developers.",
# )

# mcp_app = mcp.http_app(path="/http")
# # mcp_app = mcp.http_app(path="/sse", transport="sse")
# #
# # To test sse:
# # curl -N -H "Accept: text/event-stream" http://realm13:12345/mcp/sse
# #

# app = FastAPI(lifespan=mcp_app.lifespan)
# app.mount("/mcp", mcp_app)
app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
collections = LimitedDict(max_size=100)


empty_search_results = {
    # "distances": [dist for dist in distances if dist < threshold],
    "documents": [],
    "ids": [],
    "metadatas": [],
}


def filter_search_results_by_metadatas(
    contents, metadata_required_fields: list[str]
):
    if not metadata_required_fields:
        return contents

    has_data = ("metadatas" in contents) and (len(contents["metadatas"]) > 0)
    if not has_data:
        return empty_search_results

    print(
        f"Filtering results for required metadata fields: {metadata_required_fields}"
    )

    documents = contents["documents"]
    ids = contents["ids"]
    metadatas = contents["metadatas"]

    indices = []
    for i, metadata in enumerate(metadatas):
        if all(
            field in metadata and metadata[field]
            for field in metadata_required_fields
        ):
            indices.append(i)

    return {
        # "distances": [dist for dist in distances if dist < threshold],
        "documents": [documents[idx] for idx in indices],
        "ids": [ids[idx] for idx in indices],
        "metadatas": [metadatas[idx] for idx in indices],
    }


def unwrap_chroma_awesome_data_format(contents):
    return {
        "distances": contents["distances"][0],
        "documents": contents["documents"][0],
        "ids": contents["ids"][0],
        "metadatas": contents["metadatas"][0],
    }


def filter_search_results_by_threshold(contents, threshold: float = 0.3):
    # print(f'All (unfiltered) embeddings: {contents}')

    has_data = ("distances" in contents) and (len(contents["distances"]) > 0)
    if not has_data:
        print("Found 0 related docs.")
        return empty_search_results

    distances = contents["distances"]
    documents = contents["documents"]
    ids = contents["ids"]
    metadatas = contents["metadatas"]

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


# @app.get("/debug/mcp")
# async def debug_mcp():
#     return {"message": "MCP mount point is working", "available_at": "/mcp/sse"}


def flatten_mcp_items(data):
    grouped_by_url = {}

    for i in range(min(len(data["documents"]), len(data["metadatas"]))):
        doc_title = data["documents"][i]
        metadata = data["metadatas"][i]

        if "desc" in metadata and "url" in metadata:
            url = metadata["url"]

            if url not in grouped_by_url:
                grouped_by_url[url] = {
                    "title": doc_title,
                    "desc": metadata["desc"],
                    "url": url,
                    "date": metadata.get("date")
                    if not metadata.get("chunk")
                    else None,
                }

    result = []
    for url, item in grouped_by_url.items():
        result.append(item)

    now = datetime.datetime.now().isoformat()
    for item in result:
        if not item.get("date"):
            item["date"] = now

    return result


# @mcp.tool()
# async def devbrain_find_knowledge(q: str, token: str) -> str:
#     """Queries DevBrain (aka `developer's brain` system) and returns relevant information.

#     Args:
#         q: The question or ask to query for knowledge

#     Returns:
#         str: Helpful knowledge and context information from DevBrain (formatted as JSON list of article items, with title, short description and a URL to the original article).
#     """

#     return newsletter_find(q, token)


@app.put("/newsletter/ingest/html")
async def newsletter_ingest_html(
    item: IngestNewsletterHtml,
    background_tasks: BackgroundTasks,
    token: str = Depends(oauth2_scheme),
):
    token = access.bearer_is_valid(token)
    if not token:
        raise HTTPException(
            status_code=403, detail="Access Denied: Invalid Bearer Token"
        )

    model = mcp_provider.model_for_email(item.email_to)
    if not isinstance(model, str) or not model:
        raise HTTPException(
            status_code=400, detail=f"Invalid email_to: {item.email_to}"
        )
    neural_searcher = collections.get_or_insert(
        model,
        lambda x: neural_search.new(x),
    )
    print(f"Injesting newsletter-html for: {item.email_to} | model: {model}")

    added = mcp_provider.ingest_html(background_tasks, item, neural_searcher)
    if added == False:
        raise HTTPException(status_code=500)

    return HTMLResponse(content="<html><body>OK</body></html>", status_code=200)


@app.put("/newsletter/ingest")
def newsletter_ingest(
    item: IngestNewsletterItem, token: str = Depends(oauth2_scheme)
):
    token = access.bearer_is_valid(token)
    if not token:
        raise HTTPException(
            status_code=403, detail="Access Denied: Invalid Bearer Token"
        )

    if len(item.newsletter) == 0:
        print("Newsletter is empty. No items to ingest.")
        return HTMLResponse(
            content="<html><body>OK</body></html>", status_code=200
        )

    model = mcp_provider.model_for_email(item.email_to)
    if not isinstance(model, str) or not model:
        raise HTTPException(status_code=400, detail="Invalid email_to")
    neural_searcher = collections.get_or_insert(
        model,
        lambda x: neural_search.new(x),
    )

    added = mcp_provider.newsletter_ingest(neural_searcher, item)
    if not added:
        raise HTTPException(status_code=500)

    return HTMLResponse(content="<html><body>OK</body></html>", status_code=200)


class SearchQuery(BaseModel):
    q: str  # broad search query string
    tags: str | None = None  # comma-separated list of keywords
    n: int = 1


@app.post("/newsletter/find")
async def newsletter_find(
    query: SearchQuery,
    token: str = Depends(oauth2_scheme),
):
    token = access.bearer_is_valid(token)
    if not token:
        raise HTTPException(
            status_code=403, detail="Access Denied: Invalid Bearer Token"
        )

    tag_list = None
    if query.tags:
        tag_list = [tag.strip() for tag in query.tags.split(",") if tag.strip()]
    print(f"tag_list: {tag_list}")

    res = search(
        token, query.q, tag_list, metadata_required_fields=["desc", "url"]
    )
    items = []
    if "documents" in res and len(res["documents"]) > 0:
        items = flatten_mcp_items(res)

    print(
        f"flatten_mcp_items (drop same or incorrectly formatted): {len(res['documents'])} -> {len(items)}"
    )

    if len(items) > 0:
        return json.dumps(items, indent=2)
    else:
        return "No related knowledge at this time for this search query."


# @app.post("/search")
# def api_search(
#     query: SearchQuery,
#     token: str = Depends(oauth2_scheme),
# ):
#     if not access.bearer_is_valid(token):
#         raise HTTPException(
#             status_code=403, detail="Access Denied: Invalid Bearer Token"
#         )

#     res = search(token, query.q)
#     if "documents" in res and len(res["documents"]) > 0:
#         return res["documents"][0]
#     else:
#         return "No related knowledge."


def search(
    collection: str,
    q: str,
    tags: list[str] | None = None,
    metadata_required_fields: list[str] | None = None,
):
    print(f"Searching in c: {collection} | q: {q}")

    neural_searcher = collections.get_or_insert(
        user_modelname_to_embedding_modelname(collection),
        lambda x: neural_search.new(x),
    )

    ts = time.time()
    res = neural_searcher.search(q, tags)
    ts = time.time() - ts

    # def print_contents(contents):
    #     # print("")
    #     # print(f"filtered: {filtered}")
    #     # print(f"DOCS: {filtered['documents']}")
    #     print(
    #         f"DOCS: {len(contents['documents'])} | ids: {len(contents['ids'])} | metadatas: {len(contents['metadatas'])} | "
    #     )

    filtered = unwrap_chroma_awesome_data_format(res)
    print(
        f"Took to search using q text: {ts}s. Got {len(filtered['documents'])} results."
    )
    # print_contents(filtered)

    bn = len(filtered["documents"])
    filtered = filter_search_results_by_threshold(
        filtered,
    )
    an = len(filtered["documents"])
    print(f"Filtered by similarity threshold: {bn} -> {an}")

    # print_contents(filtered)

    if metadata_required_fields:
        bn = len(filtered["documents"])
        filtered = filter_search_results_by_metadatas(
            filtered,
            metadata_required_fields,
        )
        an = len(filtered["documents"])
        print(f"Filtered by required metadata fields: {bn} -> {an}")

    # print_contents(filtered)

    n = len(filtered["documents"])

    top_n = 10
    if n > top_n:
        ts = time.time()
        ranked = rerank.rank(q, filtered, top_n)
        ts = time.time() - ts
        print(f"Took to rerank {len(ranked['documents'])} docs: {ts}s")
        # print(f"ranked: {ranked}")
        # print_contents(ranked)

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
    token = access.bearer_is_valid(token)
    if not token:
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
    token = access.bearer_is_valid(token)
    if not token:
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
    token = access.bearer_is_valid(token)
    if not token:
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


from llm import *

llm_provider = load_module("llmprovider.yml")
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
    token = access.bearer_is_valid(token)
    if not token:
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
            collection=item.model,
            q=q,
            tags=None,
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
    token = access.bearer_is_valid(token)
    if not token:
        raise HTTPException(
            status_code=403, detail="Access Denied: Invalid Bearer Token"
        )

    return HTMLResponse(content="<html><body>OK</body></html>", status_code=200)


@app.get("/api/ps")
@app.get("/api/tags")
def ollama_models(
    token: str = Depends(oauth2_scheme),
):
    token = access.bearer_is_valid(token)
    if not token:
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
    token = access.bearer_is_valid(token)
    if not token:
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
