# RAG Project

A Retrieval-Augmented Generation (RAG) workspace that pairs document retrieval with LLM generation, and then scores the responses with both lexical and semantic metrics.

## Overview

The repository exposes two runnable scripts:
- `app.py`: Baseline `.txt` pipeline that loads articles from `news_articles/`, chunks them into overlapping windows, stores embeddings in a persistent ChromaDB, answers **"Tell me about databricks"** with `gpt-3.5-turbo`, and prints a full evaluation report against a fixed reference answer.
- `expansion_answer.py`: Experimental PDF workflow that ingests `data/microsoft-annual-report.pdf`, applies recursive + token-aware chunking via LangChain splitters, performs LLM-based query expansion before retrieval, answers with `gpt-3.5-turbo`, and computes the same suite of metrics.

Both paths enforce API-key validation, deterministic chunking, persistent (or at least replayable) vector storage, and explicit metric logging for easy comparison.

## Key RAG Concepts

### 1. Document Processing Pipeline
- **Document Loading**: `app.py` scans `.txt` files inside `news_articles/`; `expansion_answer.py` extracts text from `data/microsoft-annual-report.pdf` using `pypdf`.
- **Text Chunking**:
  - `app.py`: Naive 1000-character chunks with a 20-character overlap to preserve context.
  - `expansion_answer.py`: LangChain's recursive character splitter feeds a token-aware splitter (256 tokens, no overlap) to stay within model limits.
- **Embedding Generation**: Both scripts rely on OpenAI's `text-embedding-3-small` through Chroma's `OpenAIEmbeddingFunction`, keeping ingestion and queries aligned.

### 2. Vector Database (ChromaDB)
- **Persistent Storage**: `app.py` relies on `chromadb.PersistentClient` with the `chroma_db/` directory so embeddings survive restarts, while `expansion_answer.py` spins up an in-memory `chromadb.Client` for rapid experimentation.
- **Collection Management**: `app.py` stores chunks inside `rag_collection`; `expansion_answer.py` builds `advanced-rag-collection-openai`.
- **Embedding Function Integration**: Collections bind an OpenAI embedding function so similarity search and ingestion share the same vector space.

### 3. Similarity Search
- **Query Processing**: Converts user queries to vectors using the shared `text-embedding-3-small` encoder and cosine distance inside Chroma.
- **Query Expansion**: `expansion_answer.py` augments the user query with a hypothetical LLM-generated answer before retrieval to improve recall.
- **Ranked Retrieval**:
  - `app.py`: Returns the top 2 chunks for a concise context window.
  - `expansion_answer.py`: Returns the top 5 chunks (and optionally their embeddings) for richer downstream analysis.

### 4. Response Generation
- **Context Assembly**: Combines retrieved chunks into a coherent context.
- **Prompt Engineering**: Creates structured prompts that instruct the LLM to use retrieved context and admit uncertainty when needed.
- **Answer Synthesis**: Both scripts call the OpenAI Chat Completions API with the `gpt-3.5-turbo` model for concise, reference-backed answers.

### 5. Evaluation & Metrics
- Both scripts compute BLEU, ROUGE-1, ROUGE-L, token-level precision/recall/F1, BERTScore (precision/recall/F1), and cosine similarity between answer/reference embeddings.
- `app.py` scores against a curated Databricks–Okera paragraph; `expansion_answer.py` compares against its own query-expansion answer.

## Key Features

- **Document Processing**: Automatic loading and chunking of text documents
- **Vector Storage**: Persistent vector database with ChromaDB
- **Semantic Search**: Cosine similarity-based document retrieval
- **Context-Aware Responses**: LLM-generated answers using retrieved context
- **Similarity Scoring & Evaluation**: Transparent retrieval scores plus a full evaluation report against a reference answer
- **API Guardrails**: Each script validates `OPENAI_API_KEY` presence and `sk-` prefix before work begins

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

### Why GPT-3.5-turbo?
- **Cost Efficiency**: Affordable enough to run repeated scoring cycles during development.
- **Performance**: Adequate quality for grounded answers when supplied with retrieved context.
- **Speed**: Fast enough for interactive experimentation, including query-expansion loops.

## Setup

### Prerequisites
- Python 3.9+
- OpenAI API key that starts with `sk-`
- System tooling for `matplotlib`/`pypdf` (varies by OS)

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. Create a `news_articles` directory and add your `.txt` files (for `app.py`)
6. Place `microsoft-annual-report.pdf` under `data/` (for `expansion_answer.py`)
7. (Optional) Download NLTK data if prompted:
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

### requirements.txt

`requirements.txt` consolidates the runtime dependencies used across both scripts, including:
- `chromadb`, `openai`, `python-dotenv`
- `numpy`, `nltk`, `rouge-score`, `bert-score`
- `pypdf`, `langchain`, `sentence-transformers`, `umap-learn`, `matplotlib`

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
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (API keys)
├── .gitignore           # Git ignore rules
├── news_articles/       # Directory for input documents (.txt files)
├── data/                 # Holds microsoft-annual-report.pdf for experiments
├── chroma_db/          # ChromaDB persistent storage (auto-created)
└── venv/               # Python virtual environment
```
