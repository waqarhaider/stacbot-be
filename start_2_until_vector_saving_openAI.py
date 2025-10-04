
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from pathlib import Path
import time
import random


QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]

# --- Configuration ---

qdrant_client = QdrantClient(
    url="https://58798795-252e-43e2-b815-b0d187fb45b2.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=os.getenv("QDRANT_API_KEY"),
)

# --- Settings ---
collection_name = "stacA2_collection_1250_225_openAI"
BASE_DIR = Path(__file__).resolve().parent
PDF_FOLDER = BASE_DIR / "../stacA2pdfs"
PDF_FOLDER = PDF_FOLDER.resolve()
print(f"PDF_FOLDER set to: {PDF_FOLDER}")

# --- Load PDFs ---
documents = []
pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))

if not pdf_files:
    print(f"No PDF files found in '{PDF_FOLDER}'. Please place your PDFs there.")
else:
    for pdf_path in pdf_files:
        try:
            loader = PDFPlumberLoader(str(pdf_path))
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = str(pdf_path.name)
                documents.append(doc)
            print(f"Successfully loaded: {pdf_path.name}")
        except Exception as e:
            print(f"Error loading {pdf_path.name}: {e}")

print(f"\nTotal loaded documents from PDFs: {len(documents)}")

# --- Chunk text ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1250,
    chunk_overlap=225,
    length_function=len,
    add_start_index=True,
)
texts = text_splitter.split_documents(documents)
print(f"Split into {len(texts)} chunks.")

# --- Initialize OpenAI Embeddings ---
embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

try:
    sample_embedding = embeddings_model.embed_query("sample text")
    vector_size = len(sample_embedding)
except Exception as e:
    print(f"Error getting sample embedding size: {e}")
    exit()

print(f"OpenAI embedding vector size: {vector_size}")

# --- Create or recreate Qdrant collection ---
if qdrant_client.collection_exists(collection_name):
    qdrant_client.delete_collection(collection_name)

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

print(f"Qdrant collection '{collection_name}' recreated with vector size {vector_size}.")

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client.models import PointStruct

# --- Config ---
QDRANT_UPSERT_BATCH_SIZE = 100
QDRANT_UPSERT_DELAY_SECONDS = 5
MAX_UPSERT_RETRIES = 10
INITIAL_UPSERT_DELAY_SECONDS = 5
MAX_UPSERT_DELAY_SECONDS = 60
JITTER_RANGE = 2

MAX_EMBEDDING_RETRIES = 6
INITIAL_EMBEDDING_DELAY_SECONDS = 6
MAX_EMBEDDING_DELAY_SECONDS = 120

MAX_WORKERS = 8  # Number of parallel threads for embeddings

points_to_upsert = []

print(f"Preparing {len(texts)} chunks for embedding and upsert...")

def embed_chunk(i, text_chunk):
    """Embed a single chunk with retry logic"""
    retries = 0
    while retries < MAX_EMBEDDING_RETRIES:
        try:
            embedding = embeddings_model.embed_documents([text_chunk.page_content])[0]
            return i, text_chunk, embedding
        except Exception as e:
            retries += 1
            wait_time = INITIAL_EMBEDDING_DELAY_SECONDS * (2 ** (retries - 1)) + random.uniform(0, JITTER_RANGE)
            wait_time = min(wait_time, MAX_EMBEDDING_DELAY_SECONDS)
            print(f"Error embedding chunk {i} (Retry {retries}/{MAX_EMBEDDING_RETRIES}): {e}")
            print(f"Waiting {wait_time:.2f}s before retrying...")
            time.sleep(wait_time)
    print(f"FATAL: Failed to embed chunk {i} after {MAX_EMBEDDING_RETRIES} retries. Skipping.")
    return i, text_chunk, None

# --- Parallel embedding ---
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(embed_chunk, i, chunk) for i, chunk in enumerate(texts)]
    for future in as_completed(futures):
        i, chunk, embedding = future.result()
        if embedding is None:
            continue

        points_to_upsert.append(
            PointStruct(
                id=i,
                vector=embedding,
                payload={"page_content": chunk.page_content, "metadata": chunk.metadata},
            )
        )

        # --- Upsert in batches ---
        if len(points_to_upsert) >= QDRANT_UPSERT_BATCH_SIZE:
            upsert_retries = 0
            while upsert_retries < MAX_UPSERT_RETRIES:
                try:
                    print(f"Upserting batch of {len(points_to_upsert)} points (up to chunk {i})...")
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        wait=True,
                        points=points_to_upsert,
                    )
                    points_to_upsert = []
                    print(f"Batch upserted. Waiting {QDRANT_UPSERT_DELAY_SECONDS}s...")
                    time.sleep(QDRANT_UPSERT_DELAY_SECONDS)
                    break
                except Exception as upsert_e:
                    upsert_retries += 1
                    delay = min(INITIAL_UPSERT_DELAY_SECONDS * (2 ** (upsert_retries - 1)) + random.uniform(0, JITTER_RANGE),
                                MAX_UPSERT_DELAY_SECONDS)
                    print(f"ERROR: Failed to upsert batch (Retry {upsert_retries}/{MAX_UPSERT_RETRIES}): {upsert_e}")
                    print(f"Waiting {delay:.2f}s before retrying batch upsert...")
                    time.sleep(delay)
            else:
                print(f"FATAL: Failed to upsert batch after {MAX_UPSERT_RETRIES} retries. Skipping this batch.")
                points_to_upsert = []

# --- Final upsert for remaining points ---
if points_to_upsert:
    upsert_retries = 0
    while upsert_retries < MAX_UPSERT_RETRIES:
        try:
            print(f"Upserting final batch of {len(points_to_upsert)} points...")
            qdrant_client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points_to_upsert,
            )
            print(f"Successfully upserted final batch.")
            break
        except Exception as upsert_e:
            upsert_retries += 1
            delay = min(INITIAL_UPSERT_DELAY_SECONDS * (2 ** (upsert_retries - 1)) + random.uniform(0, JITTER_RANGE),
                        MAX_UPSERT_DELAY_SECONDS)
            print(f"ERROR: Failed to upsert final batch (Retry {upsert_retries}/{MAX_UPSERT_RETRIES}): {upsert_e}")
            print(f"Waiting {delay:.2f}s before retrying final batch...")
            time.sleep(delay)
    else:
        print(f"FATAL: Failed to upsert final batch after {MAX_UPSERT_RETRIES} retries.")

print(f"All embeddings processed and stored in Qdrant collection '{collection_name}'.")
