
import sqlite3, threading
from pydantic import BaseModel


class ModelItem(BaseModel):
    id: str
    name: str
    description: str


_connection = sqlite3.connect(
	"/data/models_mvp.db",
	check_same_thread=False,
	autocommit=True,
	# autocommit=False,
)
_connection.execute("PRAGMA journal_mode=WAL")


class DbModels: 
	def __init__(self): 
		with _connection:
		    cursor = _connection.cursor()

		    cursor.execute('CREATE TABLE IF NOT EXISTS models (id TEXT NOT NULL, name TEXT NOT NULL, description TEXT)')
		    cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_id ON models (id)')
		    _connection.commit()

		self.add_new_if_needed(
			ModelItem(
				id = 'local',
			    name = 'Local stuff',
			    description = 'Default model to save thoughts, ideas, links, etc...',
			)
		)
		self.add_new_if_needed(
			ModelItem(
				id = 'local/dev',
			    name = 'Dev',
			    description = 'Default model for code, snippets & dev guides',
			)
		)
		self.add_new_if_needed(
			ModelItem(
				id = 'tmp',
			    name = 'Blackhole',
			    description = 'This model does not save knowledge',
			)
		)

		# Reasoning / Thinking models
		self.add_new_if_needed(
			ModelItem(
				id = '*local',
			    name = '[reason-1] Local + Thinking level 1',
			    description = 'Default local + Level 1 reasoning (up to 5K thinking tokens)',
			)
		)
		self.add_new_if_needed(
			ModelItem(
				id = '**local',
			    name = '[reason-2] Local + Thinking level 2',
			    description = 'Default local + Level 2 reasoning (up to 15K thinking tokens)',
			)
		)
		self.add_new_if_needed(
			ModelItem(
				id = '***local',
			    name = '[reason-3] Local + Thinking level 3',
			    description = 'Default local + Level 3 reasoning (up to 30K thinking tokens)',
			)
		)
		self.add_new_if_needed(
			ModelItem(
				id = '*local/dev',
			    name = '[reason-1] Dev + Thinking level 1',
			    description = 'Dev + Thinking (up to 5K thinking tokens)',
			)
		)
		self.add_new_if_needed(
			ModelItem(
				id = '**local/dev',
			    name = '[reason-2] Dev + Thinking level 2',
			    description = 'Dev + Thinking (up to 15K thinking tokens)',
			)
		)
		self.add_new_if_needed(
			ModelItem(
				id = '***local/dev',
			    name = '[reason-3] Dev + Thinking level 3',
			    description = 'Dev + Thinking (up to 30K thinking tokens)',
			)
		)
		


	def list_all(self) -> list[ModelItem]:
		with _connection:
			cursor = _connection.cursor()

			rows = cursor.execute("SELECT * FROM models").fetchall()
			# print(rows)

			return map(
				lambda x: ModelItem(id=x[0], name=x[1], description=x[2]),
				rows
			)


	def update_metadata(self, item: ModelItem):
		with _connection:
			cursor = _connection.cursor()
			cursor.execute(f'UPDATE models SET name = ?, description = ? WHERE id = ?', (item.name, item.description, item.id)
			)
			_connection.commit()


	def add_new_if_needed(self, item: ModelItem) -> bool:
		with _connection:
			cursor = _connection.cursor()

			cursor.execute(f'INSERT OR IGNORE INTO models VALUES (?, ?, ?)',(item.id, item.name, item.description)
			)
			_connection.commit()
			return True


