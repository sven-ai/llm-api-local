
from llm import *
from mistralai import Mistral

class Llmprovider(LLMProviderBase):

	def __init__(self):
		api_key = os.environ.get("MISTRAL_API_KEY")

        self.client = Mistral(api_key=api_key)

	# prompt-based knowledge
	def req(
		self,
		item: ChatCompletionsItem, embeddings,
	):
		documents = embeddings['documents']
		# print(f'documents: {documents}')

		#     extra_ctx = '''
		# # Related information:
		# Use below tips to answer above question correctly.
		# '''

		#     extra_ctx = '''
		# # Helpful information:
		# Assume the verified facts below, as they may help you respond.
		# '''
		#     for idx, val in enumerate(documents):
		#         extra_ctx += f'''
		# ### Fact:
		# {val}
		#         '''
		# ### Fact ({idx + 1}):
		#     # print(f'extra_ctx: {extra_ctx}')

		docs_ctx = 'Knowledge available to you:\n====='
		for idx, val in enumerate(documents):
		    # val.replace('#', '###')
		    docs_ctx += 'Up-to-date info:\n-----\n' + val + '\n\n'


		first = item.messages[0]
		last = item.messages[-1]

		# fr = item.messages[0].role
		# fc = item.messages[0].content
		# print(f'messages[0].role: {fr}')
		# print(f'messages[0].content: {fc}')

		updated_content = None
		if len(documents) > 0:
		    if first.role == 'user':
		        # There is no `system prompt`.
		        # TODO: - check whether assistant's role is `assistant` or smth else.

		        if last.role == 'user':
		            user = ChatCompletionsPairItem(
		                role='user',
		                content=docs_ctx
		            )
		            assistant = ChatCompletionsPairItem(
		                role='assistant',
		                content='Ok, got it!'
		            )

		            # insert before `last` - before `user`.
		            item.messages.insert(-1, user)
		            item.messages.insert(-1, assistant)

		            # print(f'Added extra user/assistant convo pair:\n{docs_ctx}')
		        else:
		            print(f'OPPS. messages[0] == `user`. But messages[0] != `user`. This is not expected. Assuming bad request from the client.')
		    else:
		        # Assuming `system prompt` present.
		        item.messages[0].content = item.messages[0].content + '\n\n\n' + docs_ctx

		        # print(f'UPDATED system prompt: {item.messages[0].content}')
		else:
		    print(f'0 knowledge for the user query.')

		res = client.chat.complete(
		    # model = item.model,
		    # model = "mistral-small-latest",
		    model = remote_llm_model_for(item.model),

		    # messages = item.messages,
		    messages =
		        map(lambda x: {
		            "role": x.role,
		            "content": x.content,
		            }, item.messages)
		)

		return res, ''

		# print(chat_response.choices[0].message.content)
		# return chat_response
