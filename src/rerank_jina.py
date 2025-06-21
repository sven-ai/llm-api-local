import os

from langchain_community.document_compressors import JinaRerank

if not os.environ.get("JINA_API_KEY"):
    print("ERR. `JINA_API_KEY` env var is not set.")


class Rerank:
    def rank(self, q, contents, n=10, min_relevance=0.1):
        reranker = JinaRerank(
            model="jina-reranker-v2-base-multilingual",
            jina_api_key=os.environ.get("JINA_API_KEY"),
            top_n=n,
        )

        res0 = reranker.rerank(
            query=q,
            # documents=docs,
            documents=contents["documents"],
            top_n=n,
            model="jina-reranker-v2-base-multilingual",
        )

        # dn = len(contents['documents'])
        # # print(f'Have {dn} INPUT docs (before rerank): {contents['documents']}')
        # # print(f'RERANK res0 (unsorted): {res0}')

        res1 = filter(lambda x: x["relevance_score"] > min_relevance, res0)
        res1 = sorted(
            res1,
            key=lambda x: x["relevance_score"],
            reverse=True,  # False will sort ascending
            # Rerank returns `relevant` scores, hence bigger is better
        )
        indices = list(map(lambda x: x["index"], res1))
        # print(f"res1: {res1}")
        # print(f"indices: {indices}")

        dropped = len(res0) - len(res1)
        if dropped > 0:
            print(
                f"RERANK. Dropped {dropped} results due to low relevancy (< {min_relevance})."
            )
        # else:
        # print(f'RERANK. Sorted results: {res1}')

        documents = contents["documents"]
        ids = contents["ids"]
        metadatas = contents["metadatas"]

        # print(
        #     f"in-ranked stats 1: docs: {len(documents)} | ids: {len(ids)} | metadatas: {len(metadatas)} | "
        # )

        # print(
        #     f"in-ranked stats 2: docs: {len(documents)} | ids: {len(ids)} | metadatas: {len(metadatas)} | "
        # )

        return {
            "documents": [documents[idx] for idx in indices],
            "ids": [ids[idx] for idx in indices],
            "metadatas": [metadatas[idx] for idx in indices],
        }
