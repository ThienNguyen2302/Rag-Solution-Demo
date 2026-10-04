import os
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from ollama import Client as OllamaClient
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from turbovec.langchain import TurboQuantVectorStore

from helpers import parse_step_back_question

class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].tolist()


def generate_step_back_question(
    query: str,
    ollama_client: OllamaClient,
    model: str = "qwen3:0.6b",
) -> str:
    prompt = """
        You are an expert AI assistant using the step-back prompting technique for a
        Retrieval-Augmented Generation system about the book "The Last Lecture" by Randy Pausch.

        Given the user's specific question, write one broader conceptual question that
        captures the principles or general ideas needed to answer the original question.
        Do not answer the question. Return only the step-back question as plain text.

        Example:
        Specific question: Why did Randy Pausch create the First Penguin Award?
        Step-back question: What role does failure play in learning and personal growth?
    """.strip()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]

    response = ollama_client.chat(model=model, messages=messages)
    return parse_step_back_question(response["message"]["content"])


def generate_answer(
    query: str,
    retrieved_chunks: Sequence[str],
    ollama_client: OllamaClient,
    model: str = "qwen3:0.6b",
) -> str:
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
        You are an expert AI assistant answering questions about "The Last Lecture" by Randy Pausch.

        Answer the user's original question using only the retrieved context below.
        The context was retrieved using a broader step-back question.
        If the context does not contain enough information, say so clearly.

        Retrieved context:
        {context}
    """.strip()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]

    response = ollama_client.chat(model=model, messages=messages)
    return response["message"]["content"]


def main() -> None:
    load_dotenv()
    ollama_client = OllamaClient(host=os.getenv("OLLAMA_HOST"))

    base_dir = Path(__file__).resolve().parent
    pdf_path = base_dir / "data" / "last_lecture.pdf"
    reader = PdfReader(str(pdf_path))
    texts = [page.extract_text().strip() for page in reader.pages if page.extract_text()]

    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = character_splitter.split_text("\n\n".join(texts))

    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=0,
        tokens_per_chunk=256,
    )
    tokens = [token for chunk in chunks for token in token_splitter.split_text(chunk)]

    embedding_model = SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
    vector_store = TurboQuantVectorStore(embedding=embedding_model)
    vector_store.add_texts(tokens)

    query = input("Enter your question: ")
    step_back_question = generate_step_back_question(query, ollama_client)
    print(f"\nStep-back question: {step_back_question}")

    retrieved_chunks = [
        document.page_content
        for document in vector_store.similarity_search(step_back_question, k=5)
    ]
    final_answer = generate_answer(query, retrieved_chunks, ollama_client)
    print(f"\nFinal Answer: {final_answer}")


if __name__ == "__main__":
    main()
