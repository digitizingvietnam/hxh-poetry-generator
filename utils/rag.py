"""
rag.py

Core Retrieval-Augmented Generation (RAG) logic.

This module performs the following steps:
1. Loads a vector database (Chroma or Pinecone)
2. Performs semantic similarity search
3. Constructs a prompt using retrieved context
4. Invokes a large language model to generate a response

This module can be used by:
- Command-line scripts
- Web APIs (Flask/FastAPI)
"""

from dotenv import load_dotenv
import os
from pinecone import Pinecone
from utils.embedding import get_embedding_function
from langchain_openai import ChatOpenAI
import pandas as pd

load_dotenv()

# Pinecone configuration
PINECONE_INDEX_NAME = os.getenv("PROJECT_INDEX_NAME")
PROJECT_NAMESPACE = os.getenv("PROJECT_NAMESPACE")

# Load keywords
keyword_df = pd.read_csv("data/hxh_keywords.csv", delimiter=",")
keyword_df.columns = keyword_df.columns.str.strip()
keyword_df.columns = keyword_df.columns.str.replace("\ufeff", "")  # remove BOM


PROMPT_TEMPLATE = """
Bạn là một nhà thơ Việt Nam thế kỷ XIX, am hiểu thơ Nôm và thể Đường luật.

Dưới đây là vài bài thơ mẫu để học cấu trúc và phong cách:
{context}

---

**NHIỆM VỤ**
Hãy viết bài **vịnh "{topic}"** theo **thể thơ Nôm Đường luật** (độ dài: {num_lines} câu).

**YÊU CẦU BẮT BUỘC**
1. Phải sử dụng **ít nhất một** trong các từ/cụm từ sau đây từ bảng Hồ Xuân Hương:
{selected_keywords}

2. Giữ phong vị của Hồ Xuân Hương:
   - dân gian, phồn thực
   - mỉa mai, táo bạo mà tinh tế
   - hình ảnh sinh động, chữ Nôm gợi cảm

3. **QUAN TRỌNG**: KHÔNG viết tiêu đề hay tên chủ đề ở đầu bài thơ (như "KHOAI LANG" hay "Thơ về...").
   Chỉ viết nội dung bài thơ, bắt đầu ngay từ câu thơ đầu tiên.

4. Sau bài thơ, viết mục **CHÚ GIẢI**:
   - Mỗi từ/cụm từ được liệt kê riêng biệt, bắt đầu bằng số thứ tự (1., 2., 3.,...)
   - Mỗi từ phải có đầy đủ: Chữ Nôm, Giải cấu tạo chữ, Giải nghĩa, Trích dẫn (TV), Trích dẫn (Nôm)
   - Thay dấu '\n' trong trích dẫn (TV + Nôm) bằng dấu xuống dòng
   - Sao y nguyên văn chữ Nôm, giải cấu tạo chữ, giải nghĩa, trích dẫn (TV), trích dẫn (Nôm)

   **Định dạng chú giải:**
   1. [tên từ]
   - Chữ Nôm: [chữ Nôm]
   - Giải cấu tạo chữ: [giải thích cấu tạo]
   - Giải nghĩa: [ý nghĩa]
   - Trích dẫn (TV): [trích dẫn tiếng Việt]
   - Trích dẫn (Nôm): [trích dẫn chữ Nôm]

   2. [tên từ tiếp theo]
   - Chữ Nôm: [chữ Nôm]
   ...

---

**BẮT ĐẦU SÁNG TÁC**
"""


def format_keywords(df):
    out = ""
    for _, row in df.iterrows():
        out += (
            f"TỪ: {row['Từ / Cụm từ']}\n"
            f"Chữ Nôm: {row['Chữ Nôm']}\n"
            f"Giải cấu tạo chữ: {row['Giải cấu tạo chữ']}\n"
            f"Giải nghĩa: {row['Giải nghĩa – Thi pháp']}\n"
            f"Trích dẫn (Tiếng Việt):\n{row['Trích dẫn nguồn (Tiếng Việt)']}\n"
            f"Trích dẫn (Nôm):\n{row['Trích dẫn nguồn (Chữ Nôm)']}\n"
            "---------------------------------------\n"
        )
    return out


def query_rag(topic: str, num_lines: int = 8, num_keywords: int = 2, k: int = 4):
    """
    Query the RAG system to generate a poem.

    Args:
        topic: The subject/theme of the poem
        num_lines: Number of lines in the poem (default: 8)
        num_keywords: Number of random keywords to select (default: 2)
        k: Number of similar documents to retrieve (default: 4)

    Returns:
        dict: Contains 'poem' (generated text) and 'keywords_used' (list of keyword records)
    """
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(PINECONE_INDEX_NAME)

    # Random keyword selection
    selected = keyword_df.sample(num_keywords)
    formatted_keywords = format_keywords(selected)

    print("\n===== KEYWORDS BLOCK =====")
    print(formatted_keywords)
    print("==========================\n")

    # RAG retrieval - generate embedding for the topic
    embedding_function = get_embedding_function()
    query_embedding = embedding_function.embed_query(topic)

    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=k,
        namespace=PROJECT_NAMESPACE,
        include_metadata=True
    )

    # Extract context from results
    context_texts = []
    for match in results['matches']:
        if 'metadata' in match and 'text' in match['metadata']:
            context_texts.append(match['metadata']['text'])

    context_text = "\n---\n".join(context_texts)

    # Build prompt
    prompt = PROMPT_TEMPLATE.format(
        context=context_text,
        topic=topic,
        num_lines=num_lines,
        selected_keywords=formatted_keywords,
    )

    # Generate response
    model = ChatOpenAI(model="gpt-4o")
    answer = model.invoke(prompt).content

    return {
        "poem": answer,
        "keywords_used": selected.to_dict(orient="records")
    }
