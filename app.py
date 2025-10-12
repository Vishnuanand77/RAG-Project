import os
from dotenv import load_dotenv
import chromadb # - Open source vector database
from openai import OpenAI # - OpenAI API client
from chromadb.utils import embedding_functions # - OpenAI Embedding Function


# Load environment variables
load_dotenv()

# Get API Key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

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
    )


# Create OpenAI Client
openai_client = OpenAI(api_key=openai_api_key)

# Test the OpenAI Client
# response = openai_client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "developer", "content": "Talk like a pirate."},
#         {"role": "user", "content": "How do I check if a Python object is an instance of a class?"}
#     ]
# )

# print(response.choices[0].message.content)