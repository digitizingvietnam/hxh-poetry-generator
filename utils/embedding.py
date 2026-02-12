"""
embedding.py

Utility module for creating embedding functions used across the project.

This module is responsible for:
- Loading environment variables
- Initializing the OpenAI embedding model
- Providing a reusable embedding function for vector databases
"""

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()


def get_embedding_function():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")
    print("✅ OpenAI API key loaded.")

    return OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
