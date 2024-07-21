from langchain_text_splitters import RecursiveCharacterTextSplitter
from denser_retriever.utils import save_HF_docs_as_denser_passages, texts_to_passages
from denser_retriever.retriever_general import RetrieverGeneral
from experiments.utils_data import load_document
import logging

logger = logging.getLogger(__name__)

# Generate text chunks
file_paths = [
    "tests/test_data/state_of_the_union.txt",
    "tests/test_data/dpr.pdf",
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Build denser index
retriever_denser = RetrieverGeneral(
    "state_of_the_union", "experiments/config_local.yaml"
)


def ingest(file_path: str):
    documents = load_document(file_path)
    texts = text_splitter.split_documents(documents)
    passages = texts_to_passages(texts)
    ids = retriever_denser.ingest(passages)
    return ids


ids = []
for file_path in file_paths:
    inserted = ingest(file_path)
    ids.extend(inserted)

query = "What did the president say about Ketanji Brown Jackson"
passages, docs = retriever_denser.retrieve(query)
logger.info(passages)

# Delete the index
logger.info(len(ids))
retriever_denser.delete(ids)
