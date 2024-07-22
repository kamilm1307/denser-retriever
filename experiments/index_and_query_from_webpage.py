import logging
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from denser_retriever.retriever_general import RetrieverGeneral
from denser_retriever.utils import save_HF_docs_as_denser_passages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load, chunk and index the contents of all webpages under an url to create a retriever.
docs = WebBaseLoader("https://denser.ai").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)
passage_file = "passages.jsonl"
passages = save_HF_docs_as_denser_passages(texts, passage_file, 0)

# Build denser index
retriever_denser = RetrieverGeneral("agent_webpage", "experiments/config_local.yaml")
retriever_denser.ingest(passages)

# Query
query = "What use cases does Denser AI support?"
passages, docs = retriever_denser.retrieve(query, {})
logger.info(passages)
os.remove(passage_file)
