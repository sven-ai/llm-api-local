import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def print_embeddings(embeddings):
	# embeddings_np = embeddings.cpu().detach().numpy() 
	# np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)}) 
	# print(embeddings_np)

	# print(embeddings.float())

	print(embeddings.squeeze().tolist())



text = "what is the capital of China?"

model_path = "Alibaba-NLP/gte-modernbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Tokenize the input texts
batch_dict = tokenizer(text, max_length=8192, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = outputs.last_hidden_state[:, 0]
# # Before normalized
print_embeddings(embeddings)
 
# (Optionally) normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
# After normalized
print(embeddings.shape)
print_embeddings(embeddings)

# scores = (embeddings[:1] @ embeddings[1:].T) * 100
# print(scores.tolist())
# # [[42.89073944091797, 71.30911254882812, 33.664554595947266]]



