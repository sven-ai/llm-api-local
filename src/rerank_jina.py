
import os
from langchain_community.document_compressors import JinaRerank


if not os.environ.get("JINA_API_KEY"):
	print('ERR. `JINA_API_KEY` env var is not set.')


class Rerank:

    def rank(self, q, contents, n = 5, min_relevance = 0.1):
	    reranker = JinaRerank(
	        model="jina-reranker-v2-base-multilingual", 
	        jina_api_key=os.environ.get("JINA_API_KEY"),
	        top_n=n, 
	    )

	    res0 = reranker.rerank(
	        query=q, 
	        # documents=docs, 
	        documents=contents['documents'],
	        top_n=n, 
	        model="jina-reranker-v2-base-multilingual",
	    )

	    dn = len(contents['documents'])
	    # print(f'Have {dn} INPUT docs (before rerank): {contents['documents']}')
	    # print(f'RERANK res (unsorted): {res}')

	    res = filter(
	        lambda x: x['relevance_score'] > min_relevance,
	        res0
	    )
	    res = sorted(
	        res,
	        key=lambda x: x['relevance_score'], 
	        reverse=True # False will sort ascending
	        # Rerank returns `relevant` scores, hence bigger is better
	    )
	    indices = map(
	        lambda x: x['index'],
	        res
	    )

	    dropped = len(res0) - len(res)
	    if dropped > 0:
	        print(f'RERANK. Dropped {dropped} results due to low relevancy (< {min_relevance}).')
	    # else:
	        # print(f'RERANK. Sorted results: {res}')
	    
	    documents = contents["documents"]
	    ids = contents["ids"]
	    metadatas = contents["metadatas"]

	    return {
	        "documents": [documents[idx] for idx in indices],
	        "ids": [ids[idx] for idx in indices],
	        "metadatas": [metadatas[idx] for idx in indices],
	    }

