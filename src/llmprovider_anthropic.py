
import os
import anthropic
from llm import *
from utils import *


if not os.environ.get("ANTHROPIC_API_KEY"):
	print('ERR. `ANTHROPIC_API_KEY` env var is not set.')


def _document_custom(x) -> dict:
	return {
		"type": "document",
		"source": {
			"type": "content",
			"content": [
				{"type": "text", "text": x[1]},
				# {"type": "text", "text": "Second chunk"}
			]
		},
		"title": f'Gist / code snippet #{x[0]}',
		"context": "Source code",
		"citations": {"enabled": True}
	}

def _document_text(x) -> dict:
	return {
		"type": "document",
		"source": {
			"type": "text",
			"media_type": "text/plain",
			"data": x[1]
		},
		# "title": f'Gist / code snippet #{x[0]}',
		# "context": "Source code",
		"citations": {"enabled": True}
	}


class Llmprovider(LLMProviderBase):

	def __init__(self):
		self.client = anthropic.Anthropic()


	# Anthropic's citations-based knowledge
	def req(
		self,
		item: ChatCompletionsItem, embeddings,
	):
		client = self.client

		first = item.messages[0]
		last = item.messages[-1]

		system = ''
		if first.role == 'system':
			# Drop first message that is `system` role
			# Anthropic receives `system` as a top level param, not a role-based message contrary to openAI spec.
			system = first.content
			item.messages = item.messages[1:]

		is_dev = item.model.startswith('/dev')
		is_thinking = item.model.startswith('*')

		documents = embeddings['documents']
		# print(f'documents: {documents}')
		dn = len(documents)

		messages = list(map(
			lambda x: x.to_dict(),
			item.messages
		))

		# fr = item.messages[0].role
		# fc = item.messages[0].content
		# print(f'messages[0].role: {fr}')
		# print(f'messages[0].content: {fc}')

		if dn > 0:
			if last.role == 'user':
				content = list(map(
					lambda x: _document_custom(x) if is_dev else _document_text(x),
					enumerate(documents)
				))

				content.append(
					{'type': 'text','text': last.content}
				)

				messages[-1] = {
					'role': 'user',
					'content': content
				}

				cn = len(documents)
				print(f'Included {cn} citations as knowledge.')
			else:
				print(f'Opps. Expected last message to have a `user` role. But it is: {last.role}')
		else:
			print(f'0 knowledge for the user query.')

		if is_thinking:
			if item.model.startswith('***'):
				thinking_level = 3
			elif item.model.startswith('**'):
				thinking_level = 2
			elif item.model.startswith('*'):
				thinking_level = 1
			else:
				thinking_level = 1
			print(f'Thinking level: {thinking_level}')

			def max_tokens(tl):
				return 6000 if tl == 1 else 20000
				# return {
    #                 1: 6000,
    #                 2: 20000,
    #                 # 32K:
    #                 # anthropic-sdk dies for long contexts without streaming or batching:
    #                 # ValueError: Streaming is strongly recommended for operations that may take longer than 10 minutes. See https://github.com/anthropics/anthropic-sdk-python#long-requests for more details
    #                 # 3: 32000,
    #                 }[tl]

			def budget_tokens(tl):
				return 5000 if tl == 1 else 16000
				# return {
    #                 1: 5000,
    #                 2: 16000,
    #                 # 3: 30000,
    #                 }[tl]

			model = 'claude-3-7-sonnet-latest'
			res = client.messages.create(
     			model=model,
     			system=system,
     			messages=messages,
     			stream=False,

     			max_tokens=max_tokens(thinking_level),
                thinking={
                    "type": "enabled",
                    "budget_tokens": budget_tokens(thinking_level),
                },

     			# betas=["output-128k-2025-02-19"], # https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#extended-output-capabilities-beta
      		)
		else:
			temperature = 0.6 if is_dev else 0.9
			sonnet = 'claude-3-7-sonnet-latest'
			haiku = 'claude-3-5-haiku-latest'
			model = sonnet if is_dev else haiku
			# print(f'Forwarding to Authropic model: {model}')

			res = client.messages.create(
    			model=model,
    			system=system,
    			# TODO: - try getting these from `item` openAI request
    			# These may be set there.
    			temperature=temperature,
    			max_tokens=1000,
    			messages=messages,
    			stream=False,

    			# max_tokens=128000,
    			# betas=["output-128k-2025-02-19"], # https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#extended-output-capabilities-beta
			)
		# print(f'Anthropic response: {res}')
		print(f'Anthropic {model} usage: {res.usage}')

		# https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#implementing-extended-thinking
		def extract(type_key: str, val_key: str):
			vals = filter(
				lambda x: x.type == type_key,
				res.content
			)
			return ''.join(
				map(
					lambda x: getattr(x, val_key, ''),
					vals
				)
			)

		res_content = extract('text', 'text')
		res_thinking = extract('thinking', 'thinking')

		return res_content, res_thinking
