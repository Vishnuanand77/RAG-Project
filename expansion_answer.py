from collections import Counter
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

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

# ================================
# Evaluation Metrics
# ================================

candidate = answer.content
reference = hypothetical_answer  # Using generated hypothetical answer as reference baseline

print("\n\nEvaluation Metrics:\n")

"""
BLEU Score calculation (using 4-gram BLEU):
    - Compares 4-gram overlap between the reference and the candidate with smoothing to avoid zero scores.
    - Range: {0.0, 1.0}
        0 - Completely different
        1 - Identical

Limitations:
    - Does not consider semantic similarity but instead focuses on the exact wording.
    - Very sensitive to word choice and order.
    - Useful for machine translations.
"""
smooth = SmoothingFunction().method1
bleu = sentence_bleu(
    [reference.split()],
    candidate.split(),
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=smooth
)
bleu_percentage = bleu * 100
print(f"BLEU Score: {bleu_percentage:.2f}%")

"""
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) Score calculation (ROUGE-1 and ROUGE-L)
    - ROUGE-1: Counts unigram overlap.
    - ROUGE-L: Measures the longest common subsequence.
    - Range: {0.0, 1.0}
        0 - Completely different
        1 - Identical
Limitations:
    - Ignores semantic similarity but instead focuses on the exact wording.
    - Slightly better than BLEU as ROUGE-L accounts for matching fragments but have different meanings.
"""
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
rouge_scores = scorer.score(reference, candidate)
print(f"ROUGE-1 Score: {rouge_scores['rouge1'].fmeasure}")
print(f"ROUGE-L Score: {rouge_scores['rougeL'].fmeasure}")

""" 
Token-level Precision, Recall, F1 (counting token occurrences)
    - Precision: Measures the proportion of correctly identified tokens.
    - Recall: Measures the proportion of correctly identified tokens.
    - F1 Score: Measures the F1 score of the generated response.
    - Range: {0.0, 1.0}
        0 - Completely different
        1 - Identical
Limitations:
    - Ignores semantic similarity but instead focuses on the exact wording.
"""
ref_tokens = reference.lower().split()
cand_tokens = candidate.lower().split()
ref_counts = Counter(ref_tokens)
cand_counts = Counter(cand_tokens)
true_positive_tokens = sum((ref_counts & cand_counts).values())
precision_score = true_positive_tokens / len(cand_tokens) if cand_tokens else 0.0
recall_score = true_positive_tokens / len(ref_tokens) if ref_tokens else 0.0
f1 = (
    2 * precision_score * recall_score / (precision_score + recall_score)
    if (precision_score + recall_score) > 0
    else 0.0
)

print(f"Precision: {precision_score}")
print(f"Recall: {recall_score}")
print(f"F1 Score: {f1}")

"""
BERTScore (semantic similarity)
    - BERTScore is a semantic similarity metric that uses the BERT model to score the similarity between the reference and the candidate.
    - Each token gets a contextual embedding from BERT and then the cosine similarity is calculated between the reference and the candidate.
    - It outputs precision, recall, and F1 score.
    - Range: {-1.0, 1.0}
        -1 - Completely different
        1 - Identical
    - Captures paraphrases and synonyms better than BLEU and ROUGE.
    
    Limitations:
        - Can be slower than other metrics.
        - Requires a lot of computational resources with larger datasets.
"""
bert_p, bert_r, bert_f1 = bert_score(
    [candidate],
    [reference],
    lang="en",
    rescale_with_baseline=True
)
print(f"BERTScore Precision: {bert_p.item()}")
print(f"BERTScore Recall: {bert_r.item()}")
print(f"BERTScore F1: {bert_f1.item()}")

""" Cosine similarity between embeddings
    - Measures the cosine of the angle between two vectors.
    - Range: {-1.0, 1.0}
        -1 - Completely different
        0 - Neutral
        1 - Identical
Limitations:
    - High score does not necessarily mean high factual accuracy but it is a good measure of semantic similarity.
"""
reference_embedding = np.array(openai_embedding_function([reference])[0])
candidate_embedding = np.array(openai_embedding_function([candidate])[0])
denominator = np.linalg.norm(reference_embedding) * np.linalg.norm(candidate_embedding)
cosine_similarity = (
    float(np.dot(reference_embedding, candidate_embedding) / denominator)
    if denominator > 0
    else 0.0
)
print(f"Cosine Similarity: {cosine_similarity}")

