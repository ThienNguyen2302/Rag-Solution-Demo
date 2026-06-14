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
from sentence_transformers import SentenceTransformer
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
retrieved_chunks = [doc.page_content for doc in turbo_vec_index.similarity_search(query, k=5)]

def generate_multi_queries(query, model="qwen3:0.6b"):
    prompt = """
        You are an expert AI search assistant optimizing queries for a Retrieval-Augmented Generation (RAG) system. 
        Your specific domain of expertise is the book "The Last Lecture" by Randy Pausch.

        Your task is to take a raw user query and generate a list of 3 to 5 augmented search queries. These augmented queries will be embedded and used to search a vector database containing chunks of text from the book.

        ### Instructions:
        1. Analyze the user's core intent.
        2. Expand the query by including relevant synonyms, specific book terminology, character names (e.g., Jai, Chloe, Dylan, Logan), or core themes (e.g., brick walls, childhood dreams, time management, Tigger vs. Eeyore, overcoming obstacles).
        3. Rephrase the user's query into different formats (e.g., a direct question, a statement of concepts) to maximize the surface area for semantic matching.
        4. Correct any obvious spelling errors in the original query.
        5. Do NOT answer the user's question. ONLY output the augmented queries.

        ### Output Format:
        You must respond ONLY with a valid JSON array of strings. Do not include markdown formatting, introductory text, or explanations.

        Example Input: "what did he say about failure?"
        Example Output: 
        [
        "what did Randy Pausch say about failure and making mistakes?",
        "the concept of the First Penguin award and failing well",
        "brick walls are there to show how badly we want something",
        "learning from failure, setbacks, and overcoming obstacles in The Last Lecture"
        ]
    """.strip()

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query}
    ]

    response = ollama_client.chat(model=model, messages=messages)
    return response['message']['content']

augmented_queries = generate_multi_queries(query)
augmented_queries = json.loads(augmented_queries)

joint_queries = [query] + augmented_queries

retrieved_documents = [
    [doc.page_content for doc in turbo_vec_index.similarity_search(item, k=2)]
    for item in joint_queries
]

# deduplicate the retrieved documents
unique_documents = set()
for doc_list in retrieved_documents:
    for doc in doc_list:
        unique_documents.add(doc)

# UMAP visualization
embeddings_arr = np.asarray(embedding_model.embed_documents(token), dtype=np.float32)
original_query_emb = np.asarray(embedding_model.embed_query(query), dtype=np.float32)[None, :]
augmented_query_embs = np.asarray([embedding_model.embed_query(q) for q in augmented_queries], dtype=np.float32)

# fit UMAP (use n_components=2 for 2D)
umap_transformer = umap.UMAP(n_components=2, random_state=0)
projected_embeddings = umap_transformer.fit_transform(embeddings_arr)

# transform queries
original_query_embedding = umap_transformer.transform(original_query_emb)
augmented_query_embeddings = umap_transformer.transform(augmented_query_embs)

# create visualization
plt.figure(figsize=(12, 8))
plt.scatter(projected_embeddings[:, 0], projected_embeddings[:, 1], label='Document Embeddings', s=10, color='gray', alpha=0.6)
plt.scatter(original_query_embedding[:, 0], original_query_embedding[:, 1], label='Original Query', s=200, color='red', marker='*', edgecolors='black', linewidth=2)
plt.scatter(augmented_query_embeddings[:, 0], augmented_query_embeddings[:, 1], label='Augmented Queries', s=150, color='green', marker='o', edgecolors='black', linewidth=1.5)

plt.gca().set_aspect('equal', adjustable='datalim')
plt.title("UMAP Projection of Document and Query Embeddings", fontsize=14, fontweight='bold')
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
