import os
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

DATA_DIR="./data"
PERSIST_DIR="./storage"


def create_index():
    if not os.path.exists(PERSIST_DIR):
        # load the documents and create the index
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        # store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    return index

def query(query_engine, question):
    query_engine.query(question).print_response_stream()

def main():
    index = create_index()
    query_engine = index.as_query_engine(streaming=True)

    while True:
        query(query_engine, input("> "))
        print()

if __name__ == "__main__":
    main()
