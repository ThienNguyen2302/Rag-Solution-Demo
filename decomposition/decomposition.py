import json
import os
from pathlib import Path
from collections.abc import Sequence

from dotenv import load_dotenv
from langchain_text_splitters import (
	RecursiveCharacterTextSplitter,
	SentenceTransformersTokenTextSplitter,
)
from ollama import Client as OllamaClient
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from turbovec.langchain import TurboQuantVectorStore


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


def parse_subquestions(response: str) -> list[str]:
	subquestions = json.loads(response)
	if not isinstance(subquestions, list):
		raise ValueError("The decomposition response must be a JSON array")

	return [question.strip() for question in subquestions if isinstance(question, str) and question.strip()]


def deduplicate_documents(retrieved_documents: Sequence[Sequence[str]]) -> list[str]:
	unique_documents: list[str] = []
	seen_documents: set[str] = set()

	for documents in retrieved_documents:
		for document in documents:
			if document not in seen_documents:
				seen_documents.add(document)
				unique_documents.append(document)

	return unique_documents


def generate_subquestions(query: str, ollama_client: OllamaClient, model: str = "qwen3:0.6b") -> list[str]:
	prompt = """
		You are an expert AI assistant decomposing questions for a Retrieval-Augmented Generation system.
		The source document is the book "The Last Lecture" by Randy Pausch.

		Break the user's complex question into 2 to 4 simple, independent sub-questions.
		Each sub-question should focus on one part of the original question and contain enough context
		to be searched independently. Do not answer the questions.

		Respond ONLY with a valid JSON array of strings.
	""".strip()
	messages = [
		{"role": "system", "content": prompt},
		{"role": "user", "content": query},
	]

	response = ollama_client.chat(model=model, messages=messages)
	return parse_subquestions(response["message"]["content"])


def generate_answer(query: str, retrieved_chunks: Sequence[str], ollama_client: OllamaClient, model: str = "qwen3:0.6b") -> str:
	context = "\n\n".join(retrieved_chunks)
	prompt = f"""
		You are an expert AI assistant answering questions about "The Last Lecture" by Randy Pausch.

		Answer the user's original question using only the retrieved chunks below.
		If the chunks do not contain enough information, say so clearly.

		Retrieved chunks:
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
	subquestions = generate_subquestions(query, ollama_client)
	if not subquestions:
		subquestions = [query]

	print("\nSub-questions:")
	for subquestion in subquestions:
		print(f"- {subquestion}")

	retrieved_documents = [
		[document.page_content for document in vector_store.similarity_search(subquestion, k=3)]
		for subquestion in subquestions
	]
	retrieved_chunks = deduplicate_documents(retrieved_documents)
	final_answer = generate_answer(query, retrieved_chunks, ollama_client)
	print(f"\nFinal Answer: {final_answer}")


if __name__ == "__main__":
	main()
