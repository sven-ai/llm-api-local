import asyncio
import concurrent.futures
import json
import time
import uuid
from datetime import datetime

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from config_models import *
from kvdb import *
from loader import load_module
from models_storage import *
from utils import *

#
db_newsletters_search_query = "db_newsletters_search_query"
db_newsletters_search_tags = "db_newsletters_search_tags"
db_newsletters_search_ts = "db_newsletters_search_ts"
db_newsletters_search_rescount = "db_newsletters_search_rescount"
#
db_3designs_search_query = "db_3designs_search_query"
db_3designs_search_tags = "db_3designs_search_tags"
db_3designs_search_ts = "db_3designs_search_ts"
db_3designs_search_rescount = "db_3designs_search_rescount"
#
db_newsletters_read_url = "db_newsletters_read_url"
db_newsletters_read_ts = "db_newsletters_read_ts"
db_newsletters_read_status = "db_newsletters_read_status"

rerank = load_module("rerank.yml")
access = load_module("access.yml")
neural_search = load_module("search.yml")
web_fetch = load_module("fetch.yml")
blacklist = load_module("blacklist.yml")
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
processing_urls = set()

# Thread pool executor for background tasks
executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=10,
    thread_name_prefix="server_background",
)


empty_search_results = {
    # "distances": [dist for dist in distances if dist < threshold],
    "documents": [],
    "ids": [],
    "metadatas": [],
}


def filter_search_results_by_metadatas(contents, metadata_required_fields: list[str]):
    if not metadata_required_fields:
        return contents

    has_data = ("metadatas" in contents) and (len(contents["metadatas"]) > 0)
    if not has_data:
        return empty_search_results

    print(f"Filtering results for required metadata fields: {metadata_required_fields}")

    documents = contents["documents"]
    ids = contents["ids"]
    metadatas = contents["metadatas"]

    indices = []
    for i, metadata in enumerate(metadatas):
        if all(
            field in metadata and metadata[field] for field in metadata_required_fields
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

    indices = []
    filtered_out_docs = []
    for idx, val in enumerate(distances):
        if is_below_threshold(val):
            indices.append(idx)
        else:
            filtered_out_docs.append(documents[idx])
    if filtered_out_docs:
        print(f"Filtered out {len(filtered_out_docs)} documents due to threshold.")

    return {
        # "distances": [dist for dist in distances if dist < threshold],
        "documents": [documents[idx] for idx in indices],
        "ids": [ids[idx] for idx in indices],
        "metadatas": [metadatas[idx] for idx in indices],
    }


# @app.get("/debug/mcp")
# async def debug_mcp():
#     return {"message": "MCP mount point is working", "available_at": "/mcp/sse"}


def _is_url_blocked(
    url: str,
) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    all_blocked = blacklist.skipped_domains + blacklist.skipped_domains_post_fetch

    for blocked in all_blocked:
        if domain == blocked or domain.endswith(f".{blocked}"):
            return True

    return False


def flatten_mcp_newsletter_items(
    data,
):
    grouped_by_url = {}
    blocked_urls = []

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
                    "date": metadata.get("date") if not metadata.get("chunk") else None,
                }

    result = []
    for url, item in grouped_by_url.items():
        if _is_url_blocked(url):
            blocked_urls.append(url)
        else:
            result.append(item)

    now = current_datetime_for_sqlite()
    for item in result:
        if not item.get("date"):
            item["date"] = now

    return (
        result,
        blocked_urls,
    )


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
def newsletter_ingest(item: IngestNewsletterItem, token: str = Depends(oauth2_scheme)):
    token = access.bearer_is_valid(token)
    if not token:
        raise HTTPException(
            status_code=403, detail="Access Denied: Invalid Bearer Token"
        )

    if len(item.newsletter) == 0:
        print("Newsletter is empty. No items to ingest.")
        return HTMLResponse(content="<html><body>OK</body></html>", status_code=200)

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


def current_datetime_for_sqlite() -> str:
    """
    Returns the current date and time formatted as 'YYYY-MM-DD HH:MM:SS'
    which is a suitable string format for SQLite.
    """
    # return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return datetime.now().isoformat()


class ArticleReadQuery(BaseModel):
    url: str


async def background_fetch_and_process_article(
    url: str,
    stats_req_id: str,
):
    print("background_fetch_and_process_article")
    try:
        processing_urls.add(url)

        stats_status = kvdb_get_collection(db_newsletters_read_status)
        cache = kvdb_get_collection(db_newsletters_cache_markdown)
        raw_cache = kvdb_get_collection(db_newsletters_cache_raw_html)

        cached_raw = raw_cache.get(url)
        if cached_raw:
            cached_data = json.loads(cached_raw)
            raw_html = cached_data["html"]
            html_size = len(raw_html.encode("utf-8"))

            if html_size < 1000:
                print(f"Clearing bad cached HTML ({html_size} < 1000 bytes)")
                raw_cache.delete(url)
                fetch_result = await web_fetch.fetch(url)
                if not fetch_result:
                    stats_status.set(stats_req_id, "1")
                    return
                raw_html = fetch_result.html
            else:
                print("Using cached raw HTML for LLM processing.")
        else:
            print("Fetching raw HTML with playwright.")
            fetch_result = await web_fetch.fetch(url)
            if not fetch_result:
                stats_status.set(stats_req_id, "1")
                return
            raw_html = fetch_result.html

        html_size = len(raw_html.encode("utf-8"))
        if html_size < 1000:
            stats_status.set(stats_req_id, "1")
            return

        parsed_html = await mcp_provider.read(raw_html)

        if parsed_html and len(parsed_html) >= 100:
            cache.set(url, parsed_html)
            stats_status.set(stats_req_id, "0")
        else:
            print(
                f"⏭️  Not caching - markdown too short ({len(parsed_html) if parsed_html else 0} < 100 chars)"
            )
            stats_status.set(stats_req_id, "1")

    finally:
        processing_urls.discard(url)


def _fetch_and_process_sync(url: str, stats_req_id: str):
    """Synchronous wrapper for async background_fetch_and_process_article"""
    try:
        # print("_fetch_and_process_sync START")
        asyncio.run(background_fetch_and_process_article(url, stats_req_id))
        # print("_fetch_and_process_sync DONE")
    except Exception as e:
        print(f"_fetch_and_process_sync ERROR: {e}")
        import traceback

        traceback.print_exc()


@app.post("/newsletter/article/read")
async def newsletter_read(
    query: ArticleReadQuery,
    token: str = Depends(oauth2_scheme),
):
    token = access.bearer_is_valid(token)
    if not token:
        raise HTTPException(
            status_code=403,
            detail="Access Denied: Invalid Bearer Token",
        )

    url = normalize_url(query.url)
    if not url:
        raise HTTPException(status_code=500, detail="Cmon bro")

    if _is_url_blocked(url):
        raise HTTPException(status_code=403, detail="Domain is blocked.")

    stats_req_id = str(uuid.uuid4())
    stats_url = kvdb_get_collection(db_newsletters_read_url)
    stats_url.set(stats_req_id, url)
    stats_ts = kvdb_get_collection(db_newsletters_read_ts)
    stats_ts.set(stats_req_id, current_datetime_for_sqlite())

    cache = kvdb_get_collection(db_newsletters_cache_markdown)
    cached_parsed = cache.get(url)
    if cached_parsed:
        if len(cached_parsed) < 100:
            print(f"Clearing bad cached markdown ({len(cached_parsed)} < 100 chars)")
            cache.delete(url)
        else:
            print("Returning cached markdown.")
            stats_status = kvdb_get_collection(db_newsletters_read_status)
            stats_status.set(stats_req_id, "0")
            return cached_parsed

    if url in processing_urls:
        print(f"Article already being processed: {url}")
        raise HTTPException(
            status_code=202,
            detail="Article is being processed. Retry in a few moments.",
        )

    print(f"📖 [server] Starting article processing: {url}")
    executor.submit(
        _fetch_and_process_sync,
        url,
        stats_req_id,
    )

    raise HTTPException(
        status_code=202,
        detail="Article processing started. Retry in a few moments to retrieve content.",
    )


class SearchQuery(BaseModel):
    q: str  # broad search query string
    tags: str | None = None  # comma-separated list of keywords
    n: int = 1


@app.post("/newsletter/find_json")
async def newsletter_find_json(
    query: SearchQuery,
    token: str = Depends(oauth2_scheme),
):
    token = access.bearer_is_valid(token)
    if not token:
        raise HTTPException(
            status_code=403, detail="Access Denied: Invalid Bearer Token"
        )
    if not query.q:
        raise HTTPException(status_code=403, detail="Cmon bro")

    stats_req_id = str(uuid.uuid4())
    #
    stats_query = kvdb_get_collection(db_newsletters_search_query)
    stats_query.set(stats_req_id, query.q)
    #
    stats_tags = kvdb_get_collection(db_newsletters_search_tags)
    stats_tags.set(stats_req_id, query.tags if query.tags else "")
    #
    stats_ts = kvdb_get_collection(db_newsletters_search_ts)
    stats_ts.set(stats_req_id, current_datetime_for_sqlite())
    #

    tag_list = None
    if query.tags:
        tag_list = [tag.strip() for tag in query.tags.split(",") if tag.strip()]
    print(f"tag_list: {tag_list}")

    res = search(token, query.q, tag_list, metadata_required_fields=["desc", "url"])
    items = []
    blocked_urls = []
    if "documents" in res and len(res["documents"]) > 0:
        items, blocked_urls = flatten_mcp_newsletter_items(res)

    if blocked_urls:
        print(
            f"Found {len(blocked_urls)} blocked URLs in search results - deleting from neural search"
        )
        neural_searcher = collections.get_or_insert(
            user_modelname_to_embedding_modelname(token),
            lambda x: neural_search.new(x),
        )

        ids_to_delete = []
        for url in blocked_urls:
            ids_to_delete.append(url)
            ids_to_delete.append(f"desc+{url}")
            for format_type in ["question", "explanation"]:
                ids_to_delete.append(f"{format_type}:{url}")
                ids_to_delete.append(f"{format_type}:desc:{url}")
                ids_to_delete.append(f"{format_type}:desc+{url}")
            for fact_idx in range(20):
                ids_to_delete.append(f"fact:{fact_idx}:{url}")
                ids_to_delete.append(f"fact:{fact_idx}:desc:{url}")
                ids_to_delete.append(f"fact:{fact_idx}:desc+{url}")

        neural_searcher.delete(ids=ids_to_delete)
        print(
            f"Deleted {len(ids_to_delete)} entries for {len(blocked_urls)} blocked URLs"
        )

    n = len(items)
    print(
        f"flatten_mcp_newsletter_items (blocked: {len(blocked_urls)}, valid: {n}, total: {len(res['documents'])})"
    )

    stats_rescount = kvdb_get_collection(db_newsletters_search_rescount)
    stats_rescount.set(stats_req_id, f"{n}")

    return items


@app.post("/newsletter/find")
async def newsletter_find(
    query: SearchQuery,
    token: str = Depends(oauth2_scheme),
):
    items = await newsletter_find_json(query, token)
    n = len(items)

    if n > 0:
        if n == 1:
            return f"""
            Found 1 related article:
            ```json
            {json.dumps(items, indent=2)}
            ```

            Note: an article contains a URL to read its full content.
            """
        else:
            return f"""
            Below are {n} related articles:
            ```json
            {json.dumps(items, indent=2)}
            ```

            Note: each article contains a URL to read its full content for deeper insights.
            """
    else:
        return "No related knowledge at this time for this search query."


def flatten_mcp_3designs_items(data):
    grouped_by_code = {}

    for i in range(min(len(data["documents"]), len(data["metadatas"]))):
        contents = data["documents"][i]
        metadata = data["metadatas"][i]

        if "code" in metadata and "type" in metadata:
            code = metadata["code"]

            typeKey = {
                "krystle": "view_description",
                "view_desc": "view_description",
                "view_purpose": "view_purpose",
            }.get(metadata["type"], "view_description")
            hint = {
                "krystle": "`view_description` contains View specs (what functionality this view implements)",
                "view_desc": "`view_description` contains View description (what this view renders on screen)",
                "view_purpose": "`view_purpose` contains View purpose (what this view is best used for)",
            }.get(metadata["type"], "About this view")

            inner = grouped_by_code.get(code, [])
            inner.append(
                {
                    # "code": metadata["code"],
                    typeKey: contents,
                    # "hint": hint,
                }
            )

            grouped_by_code[code] = inner

    return grouped_by_code


@app.post("/3designs/find")
async def threedesigns_find(
    query: SearchQuery,
    token: str = Depends(oauth2_scheme),
):
    token = access.bearer_is_valid(token)
    if not token:
        raise HTTPException(
            status_code=403, detail="Access Denied: Invalid Bearer Token"
        )
    if not query.q:
        raise HTTPException(status_code=403, detail="Cmon bro")

    stats_req_id = str(uuid.uuid4())
    #
    stats_query = kvdb_get_collection(db_3designs_search_query)
    stats_query.set(stats_req_id, query.q)
    #
    stats_tags = kvdb_get_collection(db_3designs_search_tags)
    stats_tags.set(stats_req_id, query.tags if query.tags else "")
    #
    stats_ts = kvdb_get_collection(db_3designs_search_ts)
    stats_ts.set(stats_req_id, current_datetime_for_sqlite())
    #

    tag_list = None
    if query.tags:
        tag_list = [tag.strip() for tag in query.tags.split(",") if tag.strip()]
    print(f"tag_list: {tag_list}")

    res = search(token, query.q, tag_list, metadata_required_fields=["type", "code"])
    items = []
    if "documents" in res and len(res["documents"]) > 0:
        items = flatten_mcp_3designs_items(res)

    n = len(items)
    print(
        f"flatten_mcp_3designs_items (drop same or incorrectly formatted): {len(res['documents'])} -> {n}"
    )

    stats_rescount = kvdb_get_collection(db_3designs_search_rescount)
    stats_rescount.set(stats_req_id, f"{n}")

    if n > 0:
        if n == 1:
            code = next(iter(items))
            vals = items[code]

            return f"""
            Found! Unique code for the SwiftUI view: `{code}`.
            Now use `get_swiftui_view_from_3designs` tool to return full source code for it.

            Semantic information about this view:
            ```json
            {json.dumps(vals, indent=2)}
            ```

            Use code `{code}` to get this SwiftUI view, or search for another one.
            """
        else:
            return f"""
            Found {n} SwiftUI views! Check out references in the json list below (by unique code):

            ```json
            {json.dumps(items, indent=2)}
            ```

            Choose one or more views that match your requirements and use the `get_swiftui_view_from_3designs` tool to get the full SwiftUI source code.
            Use the 6-character code from the JSON above to retrieve a view.
            """
    else:
        return "Opps. We couldn't match a search term to any SwiftUI view. You may want to rephrase your query and retry. `3designs` uses semantic similarity to find the best match. Consider framing your request as view's purpose or design description. For example: view that can help increase user retention, boost brand loyalty, or minimalistic colorful view with stretchy header and sharp edges."


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

    top_n = 30
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
        description=item.metadata.get("description", f"{item.model} generic about"),
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
    return ChatCompletionsPairItem(role=x.role, content="...") if x.content == "" else x


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

    messages = list(map(lambda x: fixedChatCompletionsPairItem(x), item.messages))
    # # print(f'model: {model}')
    # print(f'fixed messages: {messages}')

    # remake with `fixed` messages - `fixed` means no message has empty content
    item.messages = messages

    if len(messages) == 0:
        return llm.plaintext_content_response(item.model, "Be nice and ❤️ Jessica.")
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
