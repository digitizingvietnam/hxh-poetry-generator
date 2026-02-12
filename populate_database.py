"""
populate_database.py

Pinecone vector database construction and update pipeline.

This script builds and maintains a Pinecone vector database for the RAG system.
It migrates data from the previous Chroma implementation to Pinecone with
improved metadata structure and namespace organization.

Pipeline:
1. Load structured data from CSV files
2. Convert rows into Document objects with metadata
3. Split documents into overlapping chunks
4. Generate embeddings for each chunk
5. Upsert embeddings into Pinecone with proper namespaces

Metadata structure:
- title: Title of the poem

Requirements:
- OPENAI_API_KEY
- PINECONE_API_KEY
- PROJECT_INDEX_NAME
- PROJECT_NAMESPACE
"""

import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from utils.embedding import get_embedding_function

load_dotenv()

DATA_PATH = "data"
PINECONE_INDEX_NAME = os.getenv("PROJECT_INDEX_NAME")
PROJECT_NAMESPACE = os.getenv("PROJECT_NAMESPACE")


def main():
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Check if index exists, if not create it
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"📦 Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # OpenAI embedding dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Change to your preferred region
            )
        )
    else:
        print(
            f"🔁 Index {PINECONE_INDEX_NAME} already exists. Clearing namespace...")
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()

        if PROJECT_NAMESPACE in stats.get("namespaces", {}):
            print(f"🧹 Clearing namespace '{PROJECT_NAMESPACE}'...")
            index.delete(delete_all=True, namespace=PROJECT_NAMESPACE)
        else:
            print(
                f"✨ Namespace '{PROJECT_NAMESPACE}' does not exist yet. Skipping delete.")

    # Get index
    index = pc.Index(PINECONE_INDEX_NAME)

    print("📘 Loading poems.csv...")
    df = pd.read_csv(os.path.join(DATA_PATH, "hxh_poems.csv"), delimiter=",")

    documents = []
    for _, row in df.iterrows():
        documents.append(
            Document(
                page_content=row["Poem"],
                metadata={"title": row["Title"]}
            )
        )

    print(f"✅ Loaded {len(documents)} poem documents")

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )

    chunks = []
    for doc in tqdm(documents, desc="Splitting documents"):
        chunks.extend(splitter.split_documents([doc]))

    print(f"✅ Created {len(chunks)} chunks")

    # Generate embeddings and prepare for Pinecone
    print("🔄 Generating embeddings and upserting to Pinecone...")
    embedding_function = get_embedding_function()

    batch_size = 100
    vectors = []

    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        # Generate embedding
        embedding = embedding_function.embed_query(chunk.page_content)

        # Prepare vector for Pinecone
        vector = {
            "id": f"poem_{i}",
            "values": embedding,
            "metadata": {
                "text": chunk.page_content,
                "title": chunk.metadata.get("title", "")
            }
        }
        vectors.append(vector)

        # Upsert in batches
        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors, namespace=PROJECT_NAMESPACE)
            vectors = []

    # Upsert remaining vectors
    if vectors:
        index.upsert(vectors=vectors, namespace=PROJECT_NAMESPACE)

    print("✅ Done! Database populated successfully.")

    # Print index stats
    stats = index.describe_index_stats()
    print(f"\n📊 Index Stats:")
    print(f"   Total vectors: {stats['total_vector_count']}")
    print(
        f"   Namespace '{PROJECT_NAMESPACE}': {stats['namespaces'].get(PROJECT_NAMESPACE, {}).get('vector_count', 0)} vectors")


if __name__ == "__main__":
    main()
