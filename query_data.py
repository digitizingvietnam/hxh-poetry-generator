"""
query_data.py

Command-line interface for querying the Retrieval-Augmented Generation (RAG) system.

This script allows users to submit a natural language query from the terminal.
The query is passed to the RAG pipeline, which retrieves relevant context from
the vector database and generates a response using a language model.

Usage:
    python query_data.py "your question here"
"""

import argparse
from utils.rag import query_rag

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str)
    args = parser.parse_args()

    print(query_rag(args.query_text))
