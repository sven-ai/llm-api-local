
from sentence_transformers import CrossEncoder

_model_path = "Alibaba-NLP/gte-reranker-modernbert-base"

# from huggingface_hub import snapshot_download
# snapshot_download(repo_id=_model_path)
# Do not use: - Donwloads more than needed for some repos


class Rerank:

    def rank(self, q, contents, n = 5, min_relevance = 0.2):
        model = CrossEncoder(
            _model_path,
            automodel_args={"torch_dtype": "auto"},
        )

        print(f'contents: {contents}')
        print(f'DOCS: {contents['documents']}')

        res0 = model.rank(
            q, contents['documents'], 
            return_documents=False, 
            top_k=n
        )
        # print(res)

        dn = len(contents['documents'])
        # print(f'Have {dn} INPUT docs (before rerank): {contents['documents']}')
        # print(f'RERANK res (unsorted): {res}')

        res = filter(
            lambda x: x['score'] > min_relevance,
            res0
        )
        res = sorted(
            res,
            key=lambda x: x['score'], 
            reverse=True # False will sort ascending
            # Rerank returns `relevant` scores, hence larger is better
        )
        indices = map(
            lambda x: x['corpus_id'],
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


