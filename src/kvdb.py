import sqlite3

from utils import *

# Stores raw fetched HTML for articles. Key: URL, Value: JSON with "html" and "date"
db_newsletters_cache_raw_html = "newsletters_cache_raw_html"
# Stores processed markdown for articles. Key: URL, Value: markdown string
db_newsletters_cache_markdown = "newsletters_cache_markdown"
# Stores incoming newsletter emails. Key: timestamp_email, Value: JSON with "html", "email_to", "date"
db_newsletters_inbox_html = "newsletters_inbox_html"
# Tracks bad article count per domain for blocklisting. Key: domain, Value: count as string
db_domain_bad_counts = "domain_bad_counts"
# Stores fetch metrics (playwright vs simple). Key: engine_status, Value: count
db_fetch_metrics = "fetch_metrics"

_connection = sqlite3.connect(
    "/data/kvs.db",
    check_same_thread=False,
    autocommit=True,
)
_connection.execute("PRAGMA journal_mode=WAL")


_collections = LimitedDict(max_size=100)


class KVCollection:
    def __init__(
        self,
        collection: str,
    ):
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

    def get(
        self,
        key: str,
    ) -> str | None:
        with _connection:
            cursor = _connection.cursor()

            rows = cursor.execute(
                f"SELECT val FROM {self.collection} WHERE key = ?", [key]
            ).fetchone()

            if rows:
                return rows[0]

            return None

    def set(
        self,
        key: str,
        val: str,
    ):
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

    def delete(
        self,
        key: str,
    ):
        with _connection:
            cursor = _connection.cursor()

            cursor.execute(
                f"DELETE FROM {self.collection} WHERE key = ?", [key]
            )
            _connection.commit()


def kvdb_get_collection(
    name: str,
) -> KVCollection:
    return _collections.get_or_insert(name, lambda x: KVCollection(name))
