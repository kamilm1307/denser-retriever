from langchain_text_splitters import RecursiveCharacterTextSplitter
from denser_retriever.utils import save_HF_docs_as_denser_passages, texts_to_passages
from denser_retriever.retriever_general import RetrieverGeneral
from experiments.utils_data import load_document
import logging
import time
import numpy as np

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
    start_time = time.time()  # Start timing the ingestion
    documents = load_document(file_path)
    texts = text_splitter.split_documents(documents)
    passages = texts_to_passages(texts)
    ids = retriever_denser.ingest(passages)
    end_time = time.time()  # End timing the ingestion
    ingest_time = end_time - start_time  # Calculate the ingest time
    logger.info(f"Ingested {len(texts)} passages from {file_path} in {ingest_time:.2f} seconds")
    return ids


ids = []
for file_path in file_paths:
    inserted = ingest(file_path)
    ids.extend(inserted)

# Run the retrieve call multiple times and calculate the average and 95th percentile time
query = "What did the president say about Ketanji Brown Jackson"
num_retrievals = 5
retrieval_times = []

for i in range(num_retrievals):
    start_time = time.time()  # Start timing the retrieval
    passages, docs = retriever_denser.retrieve(query)
    end_time = time.time()  # End timing the retrieval
    retrieval_time = end_time - start_time  # Calculate the retrieval time
    retrieval_times.append(retrieval_time)
    logger.info(f"Retrieval {i + 1}: {retrieval_time:.2f} seconds")

# Calculate average and 95th percentile
average_retrieval_time = np.mean(retrieval_times)
percentile_95_retrieval_time = np.percentile(retrieval_times, 95)

logger.info(f"Average retrieval time over {num_retrievals} runs: {average_retrieval_time:.2f} seconds")
logger.info(f"95th percentile retrieval time: {percentile_95_retrieval_time:.2f} seconds")

# Delete the index
logger.info(f"Deleting {len(ids)} passages")
retriever_denser.delete(ids)
