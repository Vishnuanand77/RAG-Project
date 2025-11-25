import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from collections import Counter
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


DOCUMENT_DIRECTORY = "news_articles"
PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "rag_collection"
DEFAULT_QUESTION = "Tell me about databricks"
REFERENCE_ANSWER = (
    "Databricks recently acquired Okera, a data governance platform with a focus on AI. "
    "They did not disclose the purchase price. Recently, Okera had raised just under "
    "$30 million from investors. Databricks emphasizes the importance of modern, AI "
    "governance solutions due to the growing volume, velocity, and variety of data across "
    "different applications."
)


# ================================
# Helpers
# ================================

def ensure_api_key() -> str:
    """
    Validate that the OpenAI API key exists and resembles the expected format.
    """
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
    if not openai_api_key.startswith("sk-"):
        raise ValueError(f"Invalid API key format. Expected to start with 'sk-', got: {openai_api_key[:10]}...")
    print(f"OpenAI API key detected: {openai_api_key[:10]}...")
    return openai_api_key


def load_documents(source_dir: str) -> List[Document]:
    """Load *.txt files from the provided directory using LangChain loaders."""
    directory = Path(source_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Document directory '{directory}' does not exist.")
    loader = DirectoryLoader(
        source_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()
    if not documents:
        raise ValueError(f"No .txt documents found under '{directory}'.")
    print(f"Loaded {len(documents)} source documents from '{directory}'.")
    return documents


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 20) -> List[Document]:
    """Split source documents into semantically overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(documents)
    print(f"Split documents into {len(splits)} chunks.")
    return splits


def prepare_vectorstore(
    embeddings: OpenAIEmbeddings,
    source_dir: str = DOCUMENT_DIRECTORY,
    persist_directory: str = PERSIST_DIRECTORY,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Create or reuse a persistent Chroma vector store populated via LangChain."""
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    existing_records = vectorstore._collection.count()  # pylint: disable=protected-access
    if existing_records > 0:
        print(f"Vector store already contains {existing_records} embeddings. Reusing persisted index.")
        return vectorstore

    documents = load_documents(source_dir)
    splits = split_documents(documents)
    vectorstore.add_documents(splits)
    vectorstore.persist()
    print(f"Persisted {len(splits)} chunks to '{persist_directory}'.")
    return vectorstore


def format_documents(documents: List[Document]) -> str:
    """Concatenate retrieved documents for prompt injection."""
    return "\n\n".join(doc.page_content for doc in documents)


def build_rag_chain(retriever, llm: ChatOpenAI):
    """Create a LangChain-style RAG pipeline from retriever to LLM."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant for question-answering tasks. Use the retrieved context to answer "
                "the question. If you don't know the answer, say that you don't know. Use at most three sentences "
                "and keep responses concise.\n\nContext:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_documents),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )
    return rag_chain


def evaluate_answer(candidate: str, reference: str, embeddings: OpenAIEmbeddings) -> Dict[str, float]:
    """Compute classic and semantic metrics for the generated answer."""
    metrics: Dict[str, float] = {}

    smooth = SmoothingFunction().method1
    bleu = sentence_bleu(
        [reference.split()],
        candidate.split(),
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth,
    )
    metrics["bleu"] = bleu * 100

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(reference, candidate)
    metrics["rouge1_f"] = rouge_scores["rouge1"].fmeasure
    metrics["rougeL_f"] = rouge_scores["rougeL"].fmeasure

    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    true_positive_tokens = sum((ref_counts & cand_counts).values())
    precision_score = true_positive_tokens / len(cand_tokens) if cand_tokens else 0.0
    recall_score = true_positive_tokens / len(ref_tokens) if ref_tokens else 0.0
    if precision_score + recall_score > 0:
        f1_score = 2 * precision_score * recall_score / (precision_score + recall_score)
    else:
        f1_score = 0.0
    metrics["precision"] = precision_score
    metrics["recall"] = recall_score
    metrics["f1"] = f1_score

    bert_p, bert_r, bert_f1 = bert_score(
        [candidate],
        [reference],
        lang="en",
        rescale_with_baseline=True,
    )
    metrics["bertscore_precision"] = bert_p.item()
    metrics["bertscore_recall"] = bert_r.item()
    metrics["bertscore_f1"] = bert_f1.item()

    reference_embedding = np.array(embeddings.embed_query(reference))
    candidate_embedding = np.array(embeddings.embed_query(candidate))
    denominator = np.linalg.norm(reference_embedding) * np.linalg.norm(candidate_embedding)
    cosine_similarity = (
        float(np.dot(reference_embedding, candidate_embedding) / denominator)
        if denominator > 0
        else 0.0
    )
    metrics["cosine_similarity"] = cosine_similarity

    return metrics


def pretty_print_metrics(metrics: Dict[str, float]) -> None:
    """Log evaluation metrics in a consistent order."""
    print("\n\nEvaluation Metrics:")
    ordered_keys = [
        "bleu",
        "rouge1_f",
        "rougeL_f",
        "precision",
        "recall",
        "f1",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
        "cosine_similarity",
    ]
    for key in ordered_keys:
        value = metrics.get(key)
        if value is not None:
            print(f" - {key}: {value:.4f}")


def main():
    api_key = ensure_api_key()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vectorstore = prepare_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    rag_chain = build_rag_chain(retriever, llm)

    answer_message = rag_chain.invoke(DEFAULT_QUESTION)
    candidate_answer = getattr(answer_message, "content", str(answer_message))
    print(f"\nQuestion: {DEFAULT_QUESTION}")
    print(f"Answer: {candidate_answer}")

    metrics = evaluate_answer(candidate_answer, REFERENCE_ANSWER, embeddings)
    pretty_print_metrics(metrics)


if __name__ == "__main__":
    main()