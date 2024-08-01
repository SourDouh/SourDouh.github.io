# import os
# import pinecone
# import sys
# import langchain
# from langchain.llms import Replicate
# from langchain.vectorstores import Pinecone
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import ConversationalRetrievalChain
# from langchain.document_loaders import WebBaseLoader

# os.environ['REPLICATE_API_TOKEN'] = "r8_55iRHnke2Y9tvgReh6YiHZtDO65bZln2IR1gH"
# pc = Pinecone(
#         api_key=os.environ.get("80eaec53-e763-4ef5-aea7-5f8b765d5708")
#     )
# #pinecone.init(api_key='80eaec53-e763-4ef5-aea7-5f8b765d5708', environment='us-east-1')

# loader = WebBaseLoader("https://sourdouh.github.io/chatbot.html")

# documents = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

# embeddings = HuggingFaceEmbeddings()

# index_name = "opus"
# index = pinecone.Index(index_name)
# vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)

# llm = Replicate(
#     model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
#     input={"temperature": 0.75, "max_length": 3000}
# )

# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm,
#     vectordb.as_retriever(search_kwargs={'k': 2}),
#     return_source_documents=True
# )

# chat_history = []
# while True:
#     query = input('Prompt: ')
#     if query == "exit" or query == "quit" or query == "q":
#         print('Exiting')
#         sys.exit()
#     result = qa_chain({'question': query, 'chat_history': chat_history})
#     print('Answer: ' + result['answer'] + '\n')
#     chat_history.append((query, result['answer']))



#----------------------------------------------------------------------------------------------------------------------------------------------




from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

from llama_index.llms.llama_cpp import LlamaCPP

# model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin"
model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    verbose=True,
)

import psycopg2

db_name = "vector_db"
host = "localhost"
password = "sourdough"
port = "5432"
user = "Joshua"
# conn = psycopg2.connect(connection_string)
conn = psycopg2.connect(
    dbname="postgres",
    host=host,
    password=password,
    port=port,
    user=user,
)
conn.autocommit = True

with conn.cursor() as c:
    c.execute(f"DROP DATABASE IF EXISTS {db_name}")
    c.execute(f"CREATE DATABASE {db_name}")

from sqlalchemy import make_url
from llama_index.vector_stores.postgres import PGVectorStore

vector_store = PGVectorStore.from_params(
    database=db_name,
    host=host,
    password=password,
    port=port,
    user=user,
    table_name="llama2_paper",
    embed_dim=384,  # openai embedding dimension
)
#load data
from pathlib import Path
from llama_index.readers.file import PyMuPDFReader

loader = PyMuPDFReader()
documents = loader.load(file_path="./data/llama2.pdf")

#text splitter to split documents
from llama_index.core.node_parser import SentenceSplitter
text_parser = SentenceSplitter(
    chunk_size=1024,
    # separator=" ",
)
text_chunks = []

# maintain relationship with source doc index, to help inject doc metadata in (3)
doc_idxs = []
for doc_idx, doc in enumerate(documents):
    cur_text_chunks = text_parser.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))

#construct nodes from text chunks

from llama_index.core.schema import TextNode

nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)

#generate embeddings for each node
for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

#add nodes to vector store
vector_store.add(nodes)


#-----------------------------------------------------------
#RETRIVAL PIPELINE
#-----------------------------------------------------------
query_str = "Can you tell me about the key concepts for safety finetuning"

#generate query emedding
query_embedding = embed_model.get_query_embedding(query_str)

#query vector database
# construct vector store query
from llama_index.core.vector_stores import VectorStoreQuery

query_mode = "default"
# query_mode = "sparse"
# query_mode = "hybrid"

vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
)

# returns a VectorStoreQueryResult
query_result = vector_store.query(vector_store_query)
print(query_result.nodes[0].get_content())

#parse result into set of nodes
from llama_index.core.schema import NodeWithScore
from typing import Optional

nodes_with_scores = []
for index, node in enumerate(query_result.nodes):
    score: Optional[float] = None
    if query_result.similarities is not None:
        score = query_result.similarities[index]
    nodes_with_scores.append(NodeWithScore(node=node, score=score))

#put into retriever
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List


class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: PGVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores

retriever = VectorDBRetriever(
    vector_store, embed_model, query_mode="default", similarity_top_k=2
)

#plug into RetrieverQueryEngine to synthesise a response
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

query_str = "How does Llama 2 perform compared to other open-source models?"

response = query_engine.query(query_str)
print(str(response))

response = query_engine.query(query_str)
print(str(response))

response = query_engine.query(query_str)
print(str(response))

response = query_engine.query(query_str)
print(str(response))

query_str = "What is the best open-source model for text to SQL?"

response = query_engine.query(query_str)
print(str(response))

response = query_engine.query(query_str)
print(str(response))

response = query_engine.query(query_str)
print(str(response))




# from llama_index import VectorStoreIndex, SimpleDirectoryReader
# documents = SimpleDirectoryReader(input_dir="/Users/Joshua/SourDouh.github.io/data").load_data()
# index = VectorStoreIndex.from_documents(documents)

# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do growing up?")
# print(response)