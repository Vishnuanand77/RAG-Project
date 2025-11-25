# RAG Project

A Retrieval-Augmented Generation (RAG) workspace that pairs document retrieval with LLM generation, and then scores the responses with classic + semantic metrics.

## Overview

The project currently ships two runnable pipelines:
- `app.py`: LangChain-based flow that ingests `.txt` news articles, persists embeddings in Chroma, answers the default question **"Tell me about databricks"** with `gpt-4o-mini`, and prints a rich evaluation report against a fixed reference answer.
- `expansion_answer.py`: An experimental notebook-style script that works directly against `data/microsoft-annual-report.pdf`, performs query expansion plus similarity search via raw Chroma, uses `gpt-3.5-turbo` for generation, and emits the same evaluation metrics.

Both paths emphasize reproducibility: API key validation, persistent vector stores, deterministic chunking, and explicit metric logging.

## Key RAG Concepts

### 1. Document Processing Pipeline
- **Document Loading**: `app.py` loads `.txt` files from the `news_articles` directory; `expansion_answer.py` extracts text from `data/microsoft-annual-report.pdf` using `pypdf`
- **Text Chunking**:
  - `app.py`: 1000-character chunks with 20-character overlap
  - `expansion_answer.py`: Recursive character splitter followed by a token-aware splitter (256 tokens, no overlap) to stay below model limits
  - *Design Decision*: Overlap preserves semantic specificity across chunks, ensuring context isn't lost at chunk boundaries
- **Embedding Generation**: Both scripts rely on OpenAI's `text-embedding-3-small` (via LangChain in `app.py`, via Chroma `OpenAIEmbeddingFunction` in `expansion_answer.py`)

### 2. Vector Database (ChromaDB)
- **Persistent Storage**: `app.py` uses LangChain's persistent `Chroma` wrapper (`chroma_db/`) to reuse embeddings across runs
- **Collection Management**: `app.py` stores chunks inside `rag_collection`; `expansion_answer.py` builds `advanced-rag-collection-openai`
- **Embedding Function Integration**: Collections bind an OpenAI embedding function so similarity search and ingestion share the same vector space

### 3. Similarity Search
- **Query Processing**: Converts user queries to vectors using the same `text-embedding-3-small` model
- **Query Expansion**: `expansion_answer.py` augments the user query with a hypothetical LLM-generated answer before retrieval to improve recall
- **Cosine Similarity**: Uses cosine similarity for better semantic matching
  - *Design Decision*: Cosine similarity ignores vector magnitude and focuses on direction, making it better for measuring meaning rather than exact wording
- **Ranked Retrieval**:
  - `app.py`: Returns top 2 chunks
  - `expansion_answer.py`: Returns top 5 chunks with optional embeddings for downstream visualization

### 4. Response Generation
- **Context Assembly**: Combines retrieved chunks into a coherent context
- **Prompt Engineering**: Creates structured prompts that instruct the LLM to use retrieved context
- **Answer Synthesis**:
  - `app.py`: `ChatOpenAI` with the `gpt-4o-mini` model (temperature 0 for determinism)
  - `expansion_answer.py`: `gpt-3.5-turbo` via the native OpenAI client

### 5. Evaluation & Metrics
- Both scripts compute BLEU, ROUGE-1, ROUGE-L, token-level precision/recall/F1, BERTScore (precision/recall/F1), and cosine similarity between answer/reference embeddings.
- `app.py` scores against a curated `REFERENCE_ANSWER` that captures the Databricks–Okera acquisition summary.
- Metric logging is centralized in `pretty_print_metrics()` for consistent ordering.

## Key Features

- **Document Processing**: Automatic loading and chunking of text documents
- **Vector Storage**: Persistent vector database with ChromaDB
- **Semantic Search**: Cosine similarity-based document retrieval
- **Context-Aware Responses**: LLM-generated answers using retrieved context
- **Similarity Scoring & Evaluation**: Transparent retrieval scores plus a full evaluation report against a reference answer
- **API Guardrails**: `ensure_api_key()` verifies the key exists and uses the `sk-` prefix before anything runs

`app.py` will:
1. Load and process documents from `news_articles/`
2. Generate embeddings and store them in ChromaDB (persisted under `chroma_db/`)
3. Execute the sample query: "Tell me about databricks"
4. Display the generated answer and a multi-metric evaluation report

`expansion_answer.py` will:
1. Parse `data/microsoft-annual-report.pdf`
2. Create overlapping + token-aware chunks and embed them into a Chroma collection
3. Expand the financial query with an LLM-generated hypothetical answer
4. Retrieve the top 5 supporting chunks, answer with GPT-3.5, and score the output

## Architecture Decisions

### Why ChromaDB?
- **Open Source**: No vendor lock-in, free to use
- **Persistence**: Data survives application restarts
- **Embedding Integration**: Built-in support for OpenAI embedding functions
- **Performance**: Optimized for similarity search operations

### Why Text Chunking with Overlap?
- **Context Preservation**: 20-character overlap ensures no semantic context is lost at boundaries
- **Optimal Size**: 1000 characters balances context richness with embedding quality
- **Retrieval Granularity**: Smaller chunks allow more precise retrieval

### Why Cosine Similarity?
- **Semantic Focus**: Measures vector direction rather than magnitude
- **Meaning Over Wording**: Better at capturing conceptual similarity
- **Robust Matching**: Less sensitive to exact word choice variations

### Why GPT Models?
- **`gpt-4o-mini` (app.py)**: Strong reasoning quality with low temperature ensures deterministic scoring runs
- **`gpt-3.5-turbo` (expansion_answer.py)**: Cost-effective for rapid query expansion and experimentation
- **Balanced Approach**: Splitting workloads keeps the LangChain demo deterministic while keeping the experimental pipeline inexpensive

## Setup

### Prerequisites
- Python 3.9+
- OpenAI API key that starts with `sk-`
- Required Python packages (see requirements below)

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install langchain langchain-openai langchain-community chromadb python-dotenv nltk rouge-score bert-score pypdf umap-learn matplotlib openai
   ```

4. Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. Create a `news_articles` directory and add your `.txt` files (for `app.py`)
6. Place `microsoft-annual-report.pdf` under `data/` (for `expansion_answer.py`)
7. (Optional) Download NLTK data if prompted: `python -c "import nltk; nltk.download('punkt')"`

### Required Dependencies
- `langchain`, `langchain-community`, `langchain-openai` - High-level orchestration for document pipelines and LLMs
- `chromadb` - Open-source vector database for similarity search
- `openai` - Native API client used by the experimental pipeline
- `python-dotenv` - Environment variable management
- `nltk`, `rouge-score`, `bert-score`, `numpy` - Evaluation metrics
- `pypdf`, `umap-learn`, `matplotlib` - PDF ingestion and (optional) embedding visualization utilities

## Usage

```bash
python app.py
# or
python expansion_answer.py
```

## Project Structure

```
RAG Project/
├── app.py                 # Main application file
├── expansion_answer.py    # Advanced PDF + query-expansion pipeline
├── README.md             # This documentation
├── .env                  # Environment variables (API keys)
├── .gitignore           # Git ignore rules
├── news_articles/       # Directory for input documents (.txt files)
├── data/                 # Holds microsoft-annual-report.pdf for experiments
├── chroma_db/          # ChromaDB persistent storage (auto-created)
└── venv/               # Python virtual environment
```
