# RAG Project

A Retrieval-Augmented Generation (RAG) application that combines document retrieval with large language model generation for intelligent question-answering.

## Overview

This RAG system processes text documents, creates vector embeddings, stores them in a vector database, and retrieves relevant context to answer user queries using OpenAI's GPT models.

## Key RAG Concepts

### 1. Document Processing Pipeline
- **Document Loading**: Loads `.txt` files from the `news_articles` directory
- **Text Chunking**: Splits documents into 1000-character chunks with 20-character overlap
  - *Design Decision*: Overlap preserves semantic specificity across chunks, ensuring context isn't lost at chunk boundaries
- **Embedding Generation**: Converts text chunks into vector representations using OpenAI's `text-embedding-3-small` model

### 2. Vector Database (ChromaDB)
- **Persistent Storage**: Uses `PersistentClient` to store data on disk for persistence across sessions
- **Collection Management**: Creates a `rag_collection` to organize document chunks
- **Embedding Function Integration**: Binds OpenAI embedding function to the collection for automatic query vectorization

### 3. Similarity Search
- **Query Processing**: Converts user queries to vectors using the same embedding function as documents
- **Cosine Similarity**: Uses cosine similarity for better semantic matching
  - *Design Decision*: Cosine similarity ignores vector magnitude and focuses on direction, making it better for measuring meaning rather than exact wording
- **Ranked Retrieval**: Returns top 2 most relevant chunks based on similarity scores

### 4. Response Generation
- **Context Assembly**: Combines retrieved chunks into a coherent context
- **Prompt Engineering**: Creates structured prompts that instruct the LLM to use retrieved context
- **Answer Synthesis**: Uses GPT-3.5-turbo to generate concise, context-aware responses

## Key Features

- **Document Processing**: Automatic loading and chunking of text documents
- **Vector Storage**: Persistent vector database with ChromaDB
- **Semantic Search**: Cosine similarity-based document retrieval
- **Context-Aware Responses**: LLM-generated answers using retrieved context
- **Similarity Scoring**: Transparent similarity scores for retrieved chunks
- **Error Handling**: API key validation and environment setup checks

The application will:
1. Load and process documents from `news_articles/`
2. Generate embeddings and store them in ChromaDB
3. Execute a sample query: "Tell me more about databricks"
4. Display similarity scores and generated response

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
- **Cost Efficiency**: More affordable than GPT-4 for response generation
- **Performance**: Sufficient quality for most RAG applications
- **Speed**: Faster response times for real-time applications

## Setup

### Prerequisites
- Python 3.9+
- OpenAI API key
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
   pip install openai chromadb python-dotenv
   ```

4. Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. Create a `news_articles` directory and add your `.txt` files

### Required Dependencies
- `openai` - OpenAI API client for embeddings and chat completions
- `chromadb` - Open source vector database for similarity search
- `python-dotenv` - Environment variable management

## Usage

```bash
python app.py
```

## Project Structure

```
RAG Project/
├── app.py                 # Main application file
├── README.md             # This documentation
├── .env                  # Environment variables (API keys)
├── .gitignore           # Git ignore rules
├── news_articles/       # Directory for input documents (.txt files)
├── chroma_db/          # ChromaDB persistent storage (auto-created)
└── venv/               # Python virtual environment
```
