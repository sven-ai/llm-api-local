import unittest
import os
from kvdb import kvdb_get_collection, KVCollection

class TestKVDB(unittest.TestCase):



    def test_get_collection(self):
        collection_name = "test_collection_1"
        collection = kvdb_get_collection(collection_name)
        self.assertIsInstance(collection, KVCollection)
        self.assertEqual(collection.collection, collection_name)

        # Test that getting the same collection name returns the same instance
        collection_2 = self.kvdb.get_collection(collection_name)
        self.assertIs(collection, collection_2)

    def test_set_and_get(self):
        collection_name = "test_collection_2"
        collection = kvdb_get_collection(collection_name)

        key1 = "key1"
        value1 = "value1"
        collection.set(key1, value1)
        self.assertEqual(collection.get(key1), value1)

        key2 = "key2"
        value2 = "value2_updated"
        collection.set(key2, "value2_old")
        collection.set(key2, value2) # Test update
        self.assertEqual(collection.get(key2), value2)

        self.assertIsNone(collection.get("non_existent_key"))

    def test_delete(self):
        collection_name = "test_collection_3"
        collection = kvdb_get_collection(collection_name)

        key1 = "key1_to_delete"
        value1 = "value1_to_delete"
        collection.set(key1, value1)
        self.assertEqual(collection.get(key1), value1)

        collection.delete(key1)
        self.assertIsNone(collection.get(key1))

        collection.delete("non_existent_key") # No return value to assert
        self.assertIsNone(collection.get("non_existent_key")) # Ensure it remains None

if __name__ == '__main__':
    unittest.main()
