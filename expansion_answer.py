from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
 
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)


import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction, OpenAIEmbeddingFunction

# ================================
# Helper Utils
# ================================

def word_wrap(text, width=87):
    """
    Wraps the given text to the specified width.

    Args:
    text (str): The text to wrap.
    width (int): The width to wrap the text to.

    Returns:
    str: The wrapped text.
    """
    return "\n".join([text[i : i + width] for i in range(0, len(text), width)])

PDF_FILE_PATH = "data/microsoft-annual-report.pdf"

# ================================
# Initizalization
# ================================

# Load environment variables
load_dotenv()

# Get API Key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is loaded and valid
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
elif not openai_api_key.startswith('sk-'):
    raise ValueError(f"Invalid API key format. Expected to start with 'sk-', got: {openai_api_key[:10]}...")

print(f"API Key loaded: {openai_api_key[:10]}...")

# Read the PDF file and extract the text
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter out empty pages
pdf_texts = [text for text in pdf_texts if text]

# Testing word wrap and text extraction
# print(word_wrap(pdf_texts[0], width=100))

# Create chunks using langchain recursive character text splitter
# Recursive Character Text Splitter:
    # Advanced text chunking algorithm that splits text while preserving semantic boundaries
    # It assigns priorities via separators and recursively tries each separator until it finds a match
    # If no separator is found, the text is split at the chunk size
    # Splits are done at natural boundaries instead of a hard limit. This way it adjusts to different document structures
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n", # Paragraph breaks
        "\n", # Line breaks
        ". ", # Full stop followed by a space
        " ", # Space
        ""], # Character-level as last resort
    chunk_size=1000,
    chunk_overlap=0
)

character_split_text = text_splitter.split_text("\n\n".join(pdf_texts))

# print(word_wrap(character_split_text[10]))
print(f"\nTotal chunks using Recursive Character Text Splitter: {len(character_split_text)}")

# Here I introduce a Token based text splitter. The reason is, we need to be mindful of model input limitations
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0,
    tokens_per_chunk=256
)

# Use token_splitter to split the text for each text in character_split_text
token_split_text = []
for text in character_split_text:
    token_split_text += token_splitter.split_text(text)

# print(word_wrap(token_split_texts[10]))
print(f"\nTotal chunks using SentenceTransformersTokenTextSplitter: {len(token_split_text)}")


# # Approach 1: Using SentenceTransformerEmbeddingFunction
# embedding_function = SentenceTransformerEmbeddingFunction()
# # print(embedding_function([token_split_texts[10]]))

# chroma_client = chromadb.Client()
# chroma_collection = chroma_client.create_collection(
#     "advanced-rag-collection-sentence-transformer", embedding_function=embedding_function
# )

# # extract the embeddings of the token_split_texts
# ids = [str(i) for i in range(len(token_split_text))]
# chroma_collection.add(ids=ids, documents=token_split_text)
# chroma_collection.count()

# Approach 2: Using OpenAIEmbeddingFunction
embedding_function = OpenAIEmbeddingFunction(api_key=openai_api_key)
# print(embedding_function([token_split_texts[10]]))

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "advanced-rag-collection-openai", embedding_function=embedding_function
)

# extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_text))]
chroma_collection.add(ids=ids, documents=token_split_text)
count = chroma_collection.count()
print(f"Total chunks in the collection: {count}")