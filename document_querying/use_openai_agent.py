from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

from llama_index.tools import QueryEngineTool
from llama_index.agent import OpenAIAgent, AgentRunner
from llama_index.tools.types import ToolMetadata
from llama_index.llms.openai import OpenAI


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
    query_engine = index.as_query_engine()
    tool = QueryEngineTool(query_engine=query_engine, metadata=ToolMetadata(name="ableton_manual", description="Ask me about the Abeton Manual"))
    agent = OpenAIAgent.from_tools(tools=[tool])

    while True:
        prompt = input("> ")
        response = agent.chat(prompt)
        print(str(response))
        print()

if __name__ == "__main__":
    main()
