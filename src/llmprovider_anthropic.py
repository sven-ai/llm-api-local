
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

		is_dev = item.model.endswith('/dev')

		temperature = 0.6 if is_dev else 0.9

		# sonnet = 'claude-3-5-sonnet-latest'
		sonnet = 'claude-3-7-sonnet-latest'
		haiku = 'claude-3-5-haiku-latest'
		model = sonnet if is_dev else haiku
		# print(f'Forwarding to Authropic model: {model}')

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


		res = client.messages.create(
			model=model,
			system=system,
			# TODO: - try getting these from `item` openAI request
			# These may be set there.
			temperature=temperature,
			max_tokens=1024,
			messages=messages,
			stream=False,

			# max_tokens=128000,
			# betas=["output-128k-2025-02-19"], # https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#extended-output-capabilities-beta
		)
		print(f'Anthropic response. Usage: {res.usage}')

		res = ''.join(
			map(
				lambda x: x.text, 
				res.content
			)
		)
		return res





