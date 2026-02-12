# Hồ Xuân Hương Poetry Generator

## Overview

An AI-powered poetry generation system that recreates the distinctive voice and style of Hồ Xuân Hương, Vietnam's most celebrated female poet from the late 18th century. This project combines literary preservation with modern AI to generate authentic Nôm poetry (chữ Nôm) in the Đường luật style, complete with traditional annotations and keyword usage from historical texts.

## Feature Highlights

- **Historical Authenticity**: Built from a curated collection of Hồ Xuân Hương's poems with proper annotations
- **Period-Accurate Style**: Replicates late 18th-century Vietnamese Nôm poetry with characteristic wit, double entendre, and social commentary
- **RAG Architecture**: Uses Retrieval-Augmented Generation for stylistically grounded poem generation based on historical examples
- **Keyword Integration**: Incorporates authentic Nôm vocabulary with detailed etymological explanations
- **Modern AI Model**: Powered by GPT-4o for optimal performance and cultural accuracy

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python web framework)
- **AI/ML**:
  - LangChain for RAG orchestration
  - OpenAI GPT-4o and embeddings (text-embedding-3-small)
  - Pinecone vector database (cloud-based)
- **Data Processing**: Custom Python scripts for poem chunking and embedding

## Technical Description

### Architecture

The poetry generator uses a Retrieval-Augmented Generation (RAG) pipeline:

```
User Topic → Flask Server → Vector Search (Pinecone) → Context Retrieval →
Keyword Selection → GPT-4o Processing → Poem Generation → Annotations → User Interface
```

### Key Components

1. **Data Pipeline** (`populate_database.py`): Processes poem corpus, generates embeddings, and populates the Pinecone vector database
2. **RAG Engine** (`utils/rag.py`): Handles query processing, semantic search, context assembly, keyword selection, and response generation
3. **Embeddings** (`utils/embedding.py`): Manages vector embeddings for documents and queries using OpenAI's embedding models
4. **Web Interface** (`main.py`): Flask REST API with session management and poem generation endpoints

### How It Works

- Historical poems are chunked and embedded into a cloud vector database (Pinecone)
- User provides a topic for the poem
- System performs semantic search to find stylistically similar poems
- Randomly selects 2 keywords from the Hồ Xuân Hương vocabulary database
- Retrieved context + keywords + prompt template is sent to GPT-4o
- The model generates a new poem in Đường luật style with detailed annotations for each keyword used

## Installation

### Prerequisites

- Python 3.8+
- pip and Git
- OpenAI API key
- Pinecone API key

### Setup Steps

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/hxh-poetry-generator.git
cd hxh-poetry-generator
```

2. **Create and activate virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure API keys**

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PROJECT_INDEX_NAME=hxh-poems-index
PROJECT_NAMESPACE=hxh-poems
```

5. **Initialize the database**

```bash
python populate_database.py
```

_Note: This may take 5-10 minutes depending on corpus size_

6. **Run the application**

```bash
python main.py
```

7. **Access at** `http://localhost:5555`

## Example Usage

**Nature Theme**

```
Request:
{
  "topic": "bầu trời",
  "num_lines": 8
}

Response:
Mây trắng bồng bềnh vẽ bầu trời,
Nắng vàng rực rỡ chiếu muôn nơi.
Chim chuyền cánh mỏi bay tìm tổ,
Gió thổi phập phòm lướt nhẹ trôi.
Mặt trời lặng lẽ ngã về chiều,
Trăng tỏ tẻo tèo teo khẽ cười.
Sao đua nhấp nháy trên trời thẳm,
Ngắm cảnh thanh bình, dạ bồi hồi.
...

CHÚ GIẢI:
1. phập phòm
- Chữ Nôm: 佛𫩓
- Giải cấu tạo chữ: 佛 (phật): mượn âm; 𫩓: ⿰口凡 (khẩu chỉ ý, phàm chỉ âm*).
- Giải nghĩa: Từ láy tượng cảm – nhịp thở, sóng động, phồn thực.
- Trích dẫn (TV): Kẽ hầm rêu mọc trơ toen miệng\nLuồng gió thông reo vỗ phập phòm.
- Trích dẫn (Nôm): 技𤞻𦼔木諸喧𠰘\n篭𩙋樁嘹撫佛𫩓
```

```
HXH-POETRY-GENERATOR/
├── data/                    # Data files
│   ├── hxh_poems.csv       # Historical poem corpus
│   └── hxh_keywords.csv    # Nôm keyword database with annotations
├── static/                  # Frontend assets (CSS, JS, images)
├── templates/               # HTML templates
│   └── index.html          # Main web interface
├── utils/                   # Core utilities
│   ├── embedding.py        # Embedding function management
│   └── rag.py              # RAG implementation and poem generation
├── main.py                  # Flask application and API routes
├── populate_database.py     # Pinecone database initialization
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

## Data Files

### hxh_poems.csv

Contains the corpus of historical Hồ Xuân Hương poems.

**Columns:**

- `Title`: Poem title
- `Poem`: Full poem text

### hxh_keywords.csv

Contains Nôm vocabulary with detailed annotations.

**Columns:**

- `Từ / Cụm từ`: Word/phrase in modern Vietnamese
- `Chữ Nôm`: Character representation in Nôm script
- `Giải cấu tạo chữ`: Character construction explanation
- `Giải nghĩa – Thi pháp`: Meaning and poetic usage
- `Trích dẫn nguồn (Tiếng Việt)`: Citation in Vietnamese
- `Trích dẫn nguồn (Chữ Nôm)`: Citation in Nôm script

## Configuration

### Environment Variables

| Variable             | Description                              | Required | Default |
| -------------------- | ---------------------------------------- | -------- | ------- |
| `OPENAI_API_KEY`     | OpenAI API key for GPT-4o and embeddings | Yes      | -       |
| `PINECONE_API_KEY`   | Pinecone API key for vector database     | Yes      | -       |
| `PROJECT_INDEX_NAME` | Name of the Pinecone index               | Yes      | -       |
| `PROJECT_NAMESPACE`  | Namespace within the Pinecone index      | Yes      | -       |
| `PORT`               | Server port                              | No       | 5555    |
