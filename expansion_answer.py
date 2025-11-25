from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
 
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from chromadb.utils import embedding_functions

import umap
import matplotlib.pyplot as plt

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

def project_embeddings(embeddings, umap_transform):
    """
    Projects the given embeddings using the provided UMAP transformer.

    Args:
    embeddings (numpy.ndarray): The embeddings to project.
    umap_transform (umap.UMAP): The trained UMAP transformer.

    Returns:
    numpy.ndarray: The projected embeddings.
    """
    projected_embeddings = umap_transform.transform(embeddings)
    return projected_embeddings

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

# Create OpenAI Client
openai_client = OpenAI(api_key=openai_api_key)

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
# Embedding Function - Allows us to create embeddings for our text data
openai_embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-small"
    )

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "advanced-rag-collection-openai", embedding_function=openai_embedding_function
)

# extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_text))]
chroma_collection.add(ids=ids, documents=token_split_text)
count = chroma_collection.count()
# print(f"Total chunks in the collection: {count}")

# Function to augment the query wit a model generated response
def augment_query_generated(query, model="gpt-3.5-turbo"):
    prompt = """You are a helpful expert financial research assistant. 
   Provide an example answer to the given question, that might be found in a document like an annual report."""
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

# # Generate a response to the query
# Use relevant chunks as context, create a prompt for the LLM, generate a response
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt,},
            {"role": "user","content": question,},
        ],
    )

    answer = response.choices[0].message
    return answer


# Query expansion using augment_query_generated
original_query = "What was the total profit for the year, and how does it compare to the previous year?"
hypothetical_answer = augment_query_generated(original_query)

joint_query = f"{original_query} {hypothetical_answer}"

# Query the collection with the joint query including documents and embeddings
results = chroma_collection.query(
    query_texts=[joint_query], 
    n_results=5, 
    include=["documents", "embeddings"]
    )

retrieved_documents = results["documents"][0]

 # Get embedding from chroma db
embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
print(f" Successfully retrieved {len(embeddings)} embeddings from ChromaDB")

# Generate a response to the query
answer = generate_response(joint_query, retrieved_documents)
print(f"\n\nAnswer: ", answer.content)