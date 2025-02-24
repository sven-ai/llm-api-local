

def user_model_name_is_valid(model_name: str) -> bool:
	parts = model_name.split('/')
	if len(parts) == 0:
		return False

	return len(parts[0]) == 20


#
# chromadb has rules for collection names:
#
# Expected collection name that (1) contains 3-63 characters, (2) starts and ends with an alphanumeric character, (3) otherwise contains only alphanumeric characters, underscores or hyphens (-), (4) contains no two consecutive periods (..) and (5) is not a valid IPv4 address, got anton/dev
#
# sven's model names are like: anton/dev
# This func makes a model-name suitable for chromadb
#
#
def user_modelname_to_embedding_modelname(name: str) -> str:
	return name.replace('/', '_')


# def remote_llm_model_for(name: str) -> str:
# 	if name.endswith('/dev'):
# 		# cannot integrate codestral into instruct-like chat
# 		# codestra does not expect `messages`
# 		# expects `prefix` & `suffix`
# 		# 
# 		# return "codestral-latest"
# 		# 
# 		return "mistral-large-latest"
# 	else:
# 		return "mistral-small-latest"


from models_storage import *

def model_config(model: ModelItem) -> dict[str, str]:
	if model.id.endswith('/dev'):
		return {
	      "id": model.id,
	      "object": "model",
	      "name": model.name + " [Dev]",
	      "description": model.description,
	      "max_context_length": 120000
		}
	else:
		return {
	      "id": model.id,
	      "object": "model",
	      "name": model.name,
	      "description": model.description,
	      "max_context_length": 120000
		}



