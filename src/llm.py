
import json, uuid, time
from pydantic import BaseModel
from pydantic_core import from_json
# from typing import Any
# from mistralai.models import ChatCompletionResponse

class ChatCompletionsPairItem(BaseModel):
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
    	return {
    		'role': self.role,
    		'content': self.content,
    	}


class ChatCompletionsItem(BaseModel):
    model: str
    messages: list[ChatCompletionsPairItem]
    stream: bool = False



class LLMProviderBase:
	def __init__(self): 
		print()


class LLM:
	def __init__(self): 
		print()


	def plaintext_content_response(
		self,
	    model: str, 
	    content: str
    ):
		res = {
			'id': str(uuid.uuid4()),
			'model': model,

			"usage": {
			    "prompt_tokens": 0,
			    "total_tokens": 0,
			    "completion_tokens": 0
			},

			'object': 'chat.completion',
			"choices": [
			    {
			        "index": 0,
			        "message": {
			            "role": "assistant",
			            "content": content
			        },
			        "finish_reason": "stop"
			    }
			]
		}

		return res

		# output = json.dumps(res)

		# return ChatCompletionResponse.model_validate(
		#     from_json(output, allow_partial=False)
		# )



	def forward_to_llm(
		self,
		provider: LLMProviderBase,
		item: ChatCompletionsItem, embeddings,
	):
		if len(item.messages) == 0:
			return self.plaintext_content_response(
	            item.model,
	            "Be nice and ❤️ Jessica."
	        )

		res = provider.req(
			item, embeddings
		)

		return self.plaintext_content_response(
			item.model,
			res
		)


	    


