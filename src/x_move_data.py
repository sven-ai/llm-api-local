from loader import load_module
from config_models import *

#
# Script moves data from chromadb's collection A -> to collection B
# and then deletes a model row from local sqlite models table.
#

name_from = user_modelname_to_embedding_modelname(
	# 'anton'
	'anton/dev'
)
name_to = user_modelname_to_embedding_modelname(
	# 'to-collection-name'
	'to-collection-name/dev'
)

def move_data():
	search = load_module('search.yml')

	m_from = search.new(name_from)
	m_to = search.new(name_to)

	items_from = m_from.peek()
	while (len(items_from) > 0) and (len(items_from['ids']) > 0):
		# print(items_from)

		ids = items_from['ids']
		embeddings = items_from['embeddings']
		documents = items_from['documents']
		metadatas = items_from['metadatas']

		for idx, doc in enumerate(documents):
			m_to.add(
				text = documents[idx],
				id = ids[idx],
		        metadata = metadatas[idx],
				embedding = embeddings[idx],
			)

		m_from.delete(ids)
		items_from = m_from.peek()


from models_storage import DbModels

def delete_model_table():
	db = DbModels()
	db.delete(name_from)



move_data()
delete_model_table()
print('All done.')
