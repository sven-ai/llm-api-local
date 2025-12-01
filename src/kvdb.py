import sqlite3

from utils import *

db_newsletters_cache_raw_html = "newsletters_cache_raw_html"
db_newsletters_inbox_html = "newsletters_inbox_html"

_connection = sqlite3.connect(
    "/data/kvs.db",
    check_same_thread=False,
    autocommit=True,
    # autocommit=False,
)
_connection.execute("PRAGMA journal_mode=WAL")


_collections = LimitedDict(max_size=100)


class KVCollection:
    def __init__(self, collection: str):
        self.collection = collection

        with _connection:
            cursor = _connection.cursor()

            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {collection} (key TEXT NOT NULL, val TEXT NOT NULL)"
            )
            cursor.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS idx_key ON {collection} (key)"
            )
            _connection.commit()

    def get(self, key: str) -> str | None:
        with _connection:
            cursor = _connection.cursor()

            rows = cursor.execute(
                f"SELECT val FROM {self.collection} WHERE key = ?", [key]
            ).fetchone()

            if rows:
                # print(
                #     f"KVDB.get {self.collection} : ({key}) -> {len(rows[0])}-bytes"
                # )
                return rows[0]

            # print(f"KVDB.get({key}) -> None")
            return None

    def set(self, key: str, val: str):
        # print(f"KVDB.set: {self.collection} : [ {key} : {len(val)}-bytes ]")

        with _connection:
            cursor = _connection.cursor()

            cursor.execute(
                f"INSERT OR IGNORE INTO {self.collection} VALUES (?, ?)",
                [key, val],
            )
            cursor.execute(
                f"UPDATE {self.collection} SET val = ? WHERE key = ?",
                [val, key],
            )

            _connection.commit()

    def delete(self, key: str):
        with _connection:
            cursor = _connection.cursor()

            cursor.execute(
                f"DELETE FROM {self.collection} WHERE key = ?", [key]
            )
            _connection.commit()


def kvdb_get_collection(name: str) -> KVCollection:
    return _collections.get_or_insert(name, lambda x: KVCollection(name))
