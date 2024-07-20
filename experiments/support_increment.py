from langchain_text_splitters import RecursiveCharacterTextSplitter
from denser_retriever.utils import save_HF_docs_as_denser_passages
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
    "state_of_the_union_5", "experiments/config_local.yaml"
)


def ingest(file_path: str):
    documents = load_document(file_path)
    texts = text_splitter.split_documents(documents)
    passage_file = "passages_" + file_path.split("/")[-1] + ".jsonl"
    passages = save_HF_docs_as_denser_passages(texts, passage_file, 0)
    logger.info(passages[0])
    retriever_denser.ingest(passages)


for file_path in file_paths:
    ingest(file_path)

query = "What did the president say about Ketanji Brown Jackson"
passages, docs = retriever_denser.retrieve(query)
logger.info(len(passages))
