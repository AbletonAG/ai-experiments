from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader,
    VectorStoreIndex,
    set_global_tokenizer,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DATA_DIR="./data"
PERSIST_DIR="./storage"

def create_index():
    if not Path(PERSIST_DIR).exists():
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

def main():

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = create_index()
    retriever = index.as_retriever()

    while True:
        nodes = retriever.retrieve(input("> "))
        for node in nodes:
            filename = node.metadata["file_name"]
            page = node.metadata["page_label"]
            print(f"Score: {node.score} File: {filename} Page: {page}")
            print(node.text)
            print("=" * 80)
        print()

if __name__ == "__main__":
    main()
