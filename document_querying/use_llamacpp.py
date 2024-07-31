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
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from transformers import AutoTokenizer


DATA_DIR="./data"
PERSIST_DIR="./storage"
MODEL_PATH="./models/llama-2-7b-chat.Q4_K_S.gguf"

MODEL_DEFAULT_OPTIONS = dict(
    temperature=0.1,
    max_new_tokens=256,
    context_window=1024,
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    model_kwargs={"n_gpu_layers": 24},
)

def load_model(**options):
    applied_options = MODEL_DEFAULT_OPTIONS | options
    return LlamaCPP(**applied_options)

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

def query(query_engine, question):
    query_engine.query(question).print_response_stream()

def main():
    set_global_tokenizer(
        AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
    )
    Settings.llm = load_model(model_path=MODEL_PATH)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    index = create_index()
    query_engine = index.as_query_engine(streaming=True)

    while True:
        query(query_engine, input("> "))
        print()

if __name__ == "__main__":
    main()
