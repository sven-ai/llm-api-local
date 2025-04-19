import time
import uuid

from pydantic import BaseModel

# from typing import Any
# from mistralai.models import ChatCompletionResponse


class ChatCompletionsPairItem(BaseModel):
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
        }


class ChatCompletionsItem(BaseModel):
    model: str
    messages: list[ChatCompletionsPairItem]
    stream: bool = False


class LLMProviderBase:
    def __init__(self):
        print()


class LLM:
    id = str(uuid.uuid4())
    at = int(time.time())

    def plaintext_content_response(
        self,
        model: str,
        content: str,
        reasoning: str = "",
        message_type: str = "message",  # Can be: `message` or `delta`
        finish_reason: str | None = "stop",  # Can be: `stop` or None
    ):
        if reasoning is not None and len(reasoning) > 0:
            m = {
                "role": "assistant",
                "content": content,
                "reasoning": reasoning,
            }
        else:
            m = {
                "role": "assistant",
                "content": content,
            }

        obj_type = (
            "chat.completion"
            if message_type == "message"
            else "chat.completion.chunk"
        )

        res = {
            "id": self.id,
            "model": model,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
                "completion_tokens": 0,
            },
            "object": obj_type,
            "created": self.at,
            "choices": [
                {"index": 0, message_type: m, "finish_reason": finish_reason}
            ],
        }

        return res

        # output = json.dumps(res)

        # return ChatCompletionResponse.model_validate(
        #     from_json(output, allow_partial=False)
        # )

    def forward_to_llm(
        self,
        provider: LLMProviderBase,
        item: ChatCompletionsItem,
        embeddings,
    ):
        if len(item.messages) == 0:
            return self.plaintext_content_response(
                item.model, "Be nice and ❤️ Jessica."
            )

        return provider.req(self, item, embeddings)
