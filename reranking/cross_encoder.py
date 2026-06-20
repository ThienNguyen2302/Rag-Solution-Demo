import os
import json
from pathlib import Path
from turbovec.langchain import TurboQuantVectorStore
import umap
import numpy as np
from pypdf import PdfReader
from ollama import Client as OllamaClient 
from dotenv import load_dotenv
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from sentence_transformers import SentenceTransformer, CrossEncoder
import matplotlib.pyplot as plt

# this class is used to load into the vector store
class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0].tolist()

load_dotenv()
key = os.getenv("OLLAMA_KEY")
host = os.getenv("OLLAMA_HOST")
ollama_client = OllamaClient(host=host)

base_dir = Path(__file__).resolve().parent
pdf_path = base_dir / "data" / "last_lecture.pdf"
reader = PdfReader(str(pdf_path))
pdf_text = [page.extract_text().strip() for page in reader.pages]
texts = [text for text in pdf_text if text]

embedding_model = SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

#split the text into smaller chunks
character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=200)
chunks = character_splitter.split_text("\n\n".join(texts))

#tokenzing the chunks using sentence transformers tokenizer
token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
token = []

for chunk in chunks:
    token += token_splitter.split_text(chunk)

turbo_vec_index = TurboQuantVectorStore(embedding=embedding_model)
turbo_vec_index.add_texts(token)

query = input("Enter your question: ")
retrieved_chunks = [doc.page_content for doc in turbo_vec_index.similarity_search(query, k=20)]

pair_query_chunk = [[query, chunk] for chunk in retrieved_chunks]
scores = cross_encoder.predict(pair_query_chunk)

# get the top 5 retrieved chunks based on the scores
top_indices = np.argsort(scores)[::-1][:5]
top_chunks = [retrieved_chunks[i] for i in top_indices]

context = "\n\n".join(top_chunks)

def generate_answer(query, retrieved_chunks):
    prompt = f"""
        You are an expert AI assistant providing answers based on retrieved information from the book "The Last Lecture" by Randy Pausch. 

        Based on the retrieved chunks of text from the book, provide a concise and accurate answer to the user's question. Use only the information contained in the retrieved chunks to formulate your response. Do not include any information that is not present in the retrieved chunks.

        ### Retrieved Chunks:
        {retrieved_chunks}

        ### Instructions:
        1. Analyze the retrieved chunks to find relevant information that directly answers the user's question.
        2. Synthesize the information into a clear and concise answer.
        3. Do NOT include any personal opinions or information that is not supported by the retrieved chunks.

        ### Output Format:
        Provide your answer as a single paragraph of text without any markdown formatting or additional explanations.
    """.strip()
    user_query = f"Based on the following retrieved chunks, answer the following question: {query}"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_query}
    ]

    response = ollama_client.chat(model="qwen3:0.6b", messages=messages)
    return response['message']['content']

final_answer = generate_answer(query, context)
print("Final Answer:", final_answer)
