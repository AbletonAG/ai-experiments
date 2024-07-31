import os
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    set_global_tokenizer,
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from transformers import AutoTokenizer


DATA_DIR="./data"
PERSIST_DIR="./storage"
MODEL_PATH="./models/llama2-7b-layla.Q4_0.gguf"

MODEL_DEFAULT_OPTIONS = dict(
    temperature=0.1,
    max_new_tokens=256,
    context_window=1024,
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

def load_model(**options):
    applied_options = MODEL_DEFAULT_OPTIONS | options
    return LlamaCPP(**applied_options)

def create_index(service_context):
    if not os.path.exists(PERSIST_DIR):
        # load the documents and create the index
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        # store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context, service_context=service_context)
    return index

def query(query_engine, question):
    query_engine.query(question).print_response_stream()

def main():
    set_global_tokenizer(
        AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
    )
    llm = load_model(model_path=MODEL_PATH)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    index = create_index(service_context)
    query_engine = index.as_query_engine(streaming=True)

    while True:
        query(query_engine, input("> "))
        print()

if __name__ == "__main__":
    main()
