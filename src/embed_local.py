
import os, time

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

_model_path = "Alibaba-NLP/gte-modernbert-base"

# from huggingface_hub import snapshot_download
# snapshot_download(repo_id=_model_path)
# Do not use: - Donwloads more than needed for some repos


class Embed:

	def __init__(self):
		self.tokenizer = AutoTokenizer.from_pretrained(_model_path)
		self.model = AutoModel.from_pretrained(_model_path)


	def text(self, text: str) -> list[float]:
		ts = time.time()

		# Tokenize the input texts
		batch_dict = self.tokenizer(text, max_length=8192, padding=True, truncation=True, return_tensors='pt')

		outputs = self.model(**batch_dict)
		embeddings = outputs.last_hidden_state[:, 0]
		# # # Before normalized
		# print_embeddings(embeddings)
		
		embeddings = F.normalize(embeddings, p=2, dim=1)
		# # After normalized
		# print(embeddings.shape)
		# print_embeddings(embeddings)

		ts = time.time() - ts
		print(f'Took to create an embedding for text: {ts}s') 

		return embeddings.squeeze().tolist()


