import chromadb
import sys

# Connection details from search_local.py
HOST = "chromadb"
PORT = 8000

# Check if the collection name is provided as an argument
if len(sys.argv) != 2:
    print("Error: Please provide the collection name as an argument.", file=sys.stderr)
    sys.exit(1)

COLLECTION_NAME = sys.argv[1]

try:
    # Initialize ChromaDB client
    client = chromadb.HttpClient(host=HOST, port=PORT)
    
    # Delete the collection
    client.delete_collection(name=COLLECTION_NAME)
    
    print(f"Successfully deleted collection: '{COLLECTION_NAME}'")

except Exception as e:
    print(f"An error occurred: {e}", file=sys.stderr)
    sys.exit(1)
