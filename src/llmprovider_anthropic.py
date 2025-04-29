import json
import os

import anthropic
from fastapi.responses import StreamingResponse

from llm import *
from utils import *

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERR. `ANTHROPIC_API_KEY` env var is not set.")


def _max_tokens(tl):
    return {
        1: 5000,
        2: 20000,
        # anthropic-sdk dies for long contexts (>32K) without streaming or batching:
        # ValueError: Streaming is strongly recommended for operations that may take longer than 10 minutes. See https://github.com/anthropics/anthropic-sdk-python#long-requests for more details
        3: 128000,
    }[tl]


def _budget_tokens(tl):
    return {
        1: 3000,
        2: 16000,
        3: 64000,
    }[tl]


def _document_custom(x) -> dict:
    return {
        "type": "document",
        "source": {
            "type": "content",
            "content": [
                {"type": "text", "text": x[1]},
                # {"type": "text", "text": "Second chunk"}
            ],
        },
        "title": f"Gist / code snippet #{x[0]}",
        "context": "Source code",
        "citations": {"enabled": True},
    }


def _document_text(x) -> dict:
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": x[1]},
        # "title": f'Gist / code snippet #{x[0]}',
        # "context": "Source code",
        "citations": {"enabled": True},
    }


def _prep_citations(
    item: ChatCompletionsItem,
    embeddings,
):
    first = item.messages[0]
    last = item.messages[-1]

    is_dev = item.model.startswith("/dev")

    system = ""
    if first.role == "system":
        # Drop first message that is `system` role
        # Anthropic receives `system` as a top level param, not a role-based message contrary to openAI spec.
        system = first.content
        item.messages = item.messages[1:]

    documents = embeddings["documents"]
    # print(f'documents: {documents}')
    dn = len(documents)

    messages = list(map(lambda x: x.to_dict(), item.messages))

    if dn > 0:
        if last.role == "user":
            content = list(
                map(
                    lambda x: _document_custom(x)
                    if is_dev
                    else _document_text(x),
                    enumerate(documents),
                )
            )

            content.append({"type": "text", "text": last.content})

            messages[-1] = {"role": "user", "content": content}

            cn = len(documents)
            print(f"Included {cn} citations as knowledge.")
        else:
            print(
                f"Opps. Expected last message to have a `user` role. But it is: {last.role}"
            )
    else:
        print("0 knowledge for the user query.")

    return system, messages


# Anthropic's citations-based proxy
class Llmprovider(LLMProviderBase):
    supportsStreaming = True

    def __init__(self):
        self.client = anthropic.Anthropic()

    def req(
        self,
        llm: LLM,
        item: ChatCompletionsItem,
        embeddings,
    ):
        self.llm = llm

        stream = item.stream and self.supportsStreaming
        print(f"Anthropic streaming: {stream}")

        if stream:
            return StreamingResponse(
                self._req_stream(item, embeddings),
                media_type="text/event-stream",
            )
        else:
            res = self._req_oneoff(item, embeddings)
            return llm.plaintext_content_response(item.model, res[0], res[1])

    def _req_oneoff(
        self,
        item: ChatCompletionsItem,
        embeddings,
    ):
        client = self.client

        system, messages = _prep_citations(
            item,
            embeddings,
        )
        is_dev = item.model.startswith("/dev")
        is_thinking = item.model.startswith("*")

        if is_thinking:
            if item.model.startswith("***"):
                thinking_level = 3
            elif item.model.startswith("**"):
                thinking_level = 2
            elif item.model.startswith("*"):
                thinking_level = 1
            else:
                thinking_level = 1
            print(f"Thinking level: {thinking_level}")

            model = "claude-3-7-sonnet-latest"
            max_tokens = _max_tokens(thinking_level)
            budget_tokens = _budget_tokens(thinking_level)
            print(f"max_tokens: {max_tokens} | budget_tokens: {budget_tokens}")

            res = client.beta.messages.create(
                model=model,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                thinking={
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                },
                betas=[
                    "output-128k-2025-02-19"
                ],  # https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#extended-output-capabilities-beta
            )
        else:
            # temperature = 0.6 if is_dev else 0.9
            sonnet = "claude-3-7-sonnet-latest"
            haiku = "claude-3-5-haiku-latest"
            model = sonnet if is_dev else haiku
            # print(f'Forwarding to Authropic model: {model}')

            res = client.messages.create(
                model=model,
                system=system,
                messages=messages,
                # TODO: - try getting these from `item` openAI request
                # These may be set there.
                # temperature=temperature,
                max_tokens=2000,
            )
        assert res is not None

        # print(f'Anthropic response: {res}')
        print(f"Anthropic usage: {res.usage}")

        # https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#implementing-extended-thinking
        def extract(type_key: str, val_key: str):
            vals = filter(lambda x: x.type == type_key, res.content)
            return "".join(map(lambda x: getattr(x, val_key, ""), vals))

        res_content = extract("text", "text")
        res_thinking = extract("thinking", "thinking")

        return res_content, res_thinking

    def _req_stream(
        self,
        item: ChatCompletionsItem,
        embeddings,
    ):
        client = self.client

        system, messages = _prep_citations(
            item,
            embeddings,
        )
        is_dev = item.model.startswith("/dev")
        is_thinking = item.model.startswith("*")

        if is_thinking:
            if item.model.startswith("***"):
                thinking_level = 3
            elif item.model.startswith("**"):
                thinking_level = 2
            elif item.model.startswith("*"):
                thinking_level = 1
            else:
                thinking_level = 1
            print(f"Thinking level: {thinking_level}")

            model = "claude-3-7-sonnet-latest"
            max_tokens = _max_tokens(thinking_level)
            budget_tokens = _budget_tokens(thinking_level)
            print(f"max_tokens: {max_tokens} | budget_tokens: {budget_tokens}")

            res = client.beta.messages.stream(
                model=model,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                thinking={
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                },
                betas=[
                    "output-128k-2025-02-19"
                ],  # https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#extended-output-capabilities-beta
            )
        else:
            # temperature = 0.6 if is_dev else 0.9
            sonnet = "claude-3-7-sonnet-latest"
            haiku = "claude-3-5-haiku-latest"
            model = sonnet if is_dev else haiku
            # print(f'Forwarding to Authropic model: {model}')

            res = client.messages.stream(
                model=model,
                system=system,
                messages=messages,
                # TODO: - try getting these from `item` openAI request
                max_tokens=2000,
            )
        assert res is not None

        # print(f'RES: {res}')
        with res as stream:
            for x in stream:  # .text_stream:
                # print('CHUNK:')
                # print(x)

                if (x.type == "text") or (x.type == "thinking"):
                    res_content = getattr(
                        x, "text", ""
                    )  # extract("text", "text")
                    res_thinking = getattr(
                        x, "thinking", ""
                    )  # extract("thinking", "thinking")

                    # print(f'Yield: {res_content} | {res_thinking}')
                    contents = json.dumps(
                        self.llm.plaintext_content_response(
                            item.model,
                            res_content,
                            res_thinking,
                            message_type="delta",
                            finish_reason=None,
                        )
                    )
                    yield f"event: message\ndata: {contents}\n\n"
                elif x.type == "message_stop":
                    # elif x.type == 'content_block_stop':
                    # print("Anthropic - streaming response ended.")

                    contents = json.dumps(
                        self.llm.plaintext_content_response(
                            item.model,
                            "",
                            message_type="delta",
                            finish_reason="stop",
                        )
                    )
                    yield f"event: message\ndata: {contents}\n\n"
                # else:
                #     print(f"Anthropic parsing: skipped chunk type: {x.type}")
