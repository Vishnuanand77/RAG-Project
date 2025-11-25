import os
from dotenv import load_dotenv
import chromadb # - Open source vector database
from openai import OpenAI # - OpenAI API client
from chromadb.utils import embedding_functions # - OpenAI Embedding Function

# To calculate BLEU and ROUGE scores, we'll use external libraries: nltk for BLEU and rouge_score for ROUGE.
# For precision, recall, and F1, we'll use sklearn's classification_report utilities for a simple token-level metric.
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support

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

# Embedding Function - Allows us to create embeddings for our text data
openai_embedding_functions = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-small"
    )

# Initialize ChromaDB Client with Persistence
# Persistence means that the data is stored on the disk, Other methods include :
# Client() - In-memory client
# HttpClient() - Client that uses HTTP requests to interact with a remote database
chroma_client = chromadb.PersistentClient(
    path="chroma_db"
    )

# Create a Collection
# Collection is a table or a group of documents in a database
collection = chroma_client.get_or_create_collection(
    name="rag_collection",
    embedding_function=openai_embedding_functions
    # Use cosine similarity instead of euclidean distance
        # Cosine is better as it ignores the magnitude of the vectors and instead measure direction of the vectors
        # Focuses more on the meaning instead of the exact wording
    # metadata={"hnsw:space": "cosine"}
    )


# ================================
# Functions
# ================================

# Test the OpenAI Client
def test_openai_client():
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "Talk like a pirate."},
            {"role": "user", "content": "How do I check if a Python object is an instance of a class?"}
        ]
    )
    
    print(response.choices[0].message.content)

# Function to load documents from a directory
def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

# Function to create chunks of documents, chunk size 1000, chunk overlap 20
# We add overlap to preserve semantic specificity of the documents across chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Function to create embeddings using OpenAI
def create_embeddings(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# ================================
# Main Execution
# ================================

# Create OpenAI Client
openai_client = OpenAI(api_key=openai_api_key)

# Load documents from news_articles directory
documents = load_documents("news_articles")
print("Loaded documents: ", len(documents))

# Split documents into chunks
chunked_documents = []
for document in documents:
    chunks = split_text(document["text"])
    print("Split document into chunks: ", len(chunks))
    for i, chunk in enumerate(chunks):
        chunked_documents.append({
            "id": f"{document['id']}_chunk{i+1}", 
            "text": chunk}
            )
# Print length of chunked documents
# print("Length of chunked documents: ", len(chunked_documents))

# Generate embeddings for chunked documents
for document in chunked_documents:
    # Logging the document text
    print(f"Generating embeddings for document: {document['id']}")
    document["embeddings"] = create_embeddings(document["text"])

# print(document["embeddings"])

# Upsert chunked documents into ChromaDB for each chunk and its chunk embedding
for document in chunked_documents:
    collection.upsert(
        ids=[document["id"]],
        documents=[document["text"]],
        embeddings=[document["embeddings"]]
    )

# Function to query the collection and extract relevant chunks
# Print the similarity score of the query and the relevant chunks
def query_collection(query):

    # The query function performs the similarity search
    # How it works:
        # Query gets converted to a vector using the same embedding function as the documents
        # ChromaDB calculates similarity between the query vector and the document vectors
        # Documents are ranked by similarity score
        # n_results is the number of documents to return (Default is 10)
    # Since we are using "collection" it binds the embedding function to the query
    results = collection.query(query_texts=[query], n_results=2)
    
    relevant_chunks = []
    for i, (doc, score) in enumerate(zip(results["documents"][0], results["distances"][0])):
        print(f"Chunk {i+1} Similarity Score: {score}")
        print(f"Text: {doc}\n")
        relevant_chunks.append(doc)
    return relevant_chunks

# Generate a response to the query
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


# ================================
# Example Usage
# ================================

# Query the collection
question = "Tell me about databricks"
results = query_collection(question)
print("Query results: ", results)

# Generate a response to the query
answer = generate_response(question, results)
print(f"\n\nAnswer: ", answer.content)

# ================================
# Evaluation
# ================================

# Evaluate the response using RAG Evaluation Metrics - BLEU Score, ROUGE Score, Precision, Recall, F1 Score
# BLEU Score - Measures the similarity between the generated response and the reference answer
# ROUGE Score - Measures the similarity between the generated response and the reference answer
# Precision - Measures the precision of the generated response
# Recall - Measures the recall of the generated response
# F1 Score - Measures the F1 score of the generated response

# Ensure reference_answer and answer.content are strings
candidate = answer.content
reference = "Databricks has acquired Okera, an AI-centric data governance platform, to strengthen its ability to manage and secure data and AI assets at scale, particularly as the rise of large language models accelerates the creation of machine-generated data and exposes the limits of traditional, centralized governance controls. Okera provides AI-powered automatic discovery and classification of sensitive information, metadata-driven policy tagging, and an isolation technology that can enforce governance on arbitrary workloads with minimal overhead—capabilities that Databricks plans to integrate directly into its Unity Catalog to enhance unified governance across clouds and workloads. The acquisition also brings Okera co-founder and CEO Nong Li—creator of Apache Parquet and former Databricks engineer—back to the company, as Databricks aims to offer modern governance APIs and support enterprises that struggle to manage access policies amid rapidly growing data volume, velocity, and LLM-driven complexity."

# BLEU Score calculation (using unigram BLEU for simplicity)
smooth = SmoothingFunction().method1
bleu = sentence_bleu([reference.split()], candidate.split(), smoothing_function=smooth)
print(f"BLEU Score: {bleu}")

# ROUGE Score calculation (ROUGE-1 and ROUGE-L)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
rouge_scores = scorer.score(reference, candidate)
print(f"ROUGE-1 Score: {rouge_scores['rouge1'].fmeasure}")
print(f"ROUGE-L Score: {rouge_scores['rougeL'].fmeasure}")

# Token-level Precision, Recall, F1 (based on overlap of unique tokens)
ref_tokens = set(reference.lower().split())
cand_tokens = set(candidate.lower().split())
true_positives = len(ref_tokens & cand_tokens)
precision_score = true_positives / (len(cand_tokens) if len(cand_tokens) > 0 else 1)
recall_score = true_positives / (len(ref_tokens) if len(ref_tokens) > 0 else 1)
f1 = 2 * precision_score * recall_score / (precision_score + recall_score) if (precision_score + recall_score) else 0

print(f"Precision: {precision_score}")
print(f"Recall: {recall_score}")
print(f"F1 Score: {f1}")