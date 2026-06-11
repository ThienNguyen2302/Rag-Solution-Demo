import os
import chromadb
import umap
import numpy as np
from pypdf import PdfReader
from ollama import Client as OllamaClient 
from dotenv import load_dotenv
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import matplotlib.pyplot as plt

load_dotenv()
key = os.getenv("OLLAMA_KEY")
host = os.getenv("OLLAMA_HOST")
ollama_client = OllamaClient(host=host)

reader = PdfReader("data/last_lecture.pdf")
pdf_text = [page.extract_text().strip() for page in reader.pages]
texts = [text for text in pdf_text if text]

#split the text into smaller chunks
character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=200)
chunks = character_splitter.split_text("\n\n".join(texts))

# print(f"Number of chunks: {len(chunks)}")
# print(f"First chunk: {chunks[0]}")

#tokenzing the chunks using sentence transformers tokenizer
token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
token = []

for chunks in chunks:
    token += token_splitter.split_text(chunks)

# print(f"Number of tokenized chunks: {len(token)}")
# print(f"First tokenized chunk: {token[0]}")

embedding_function = SentenceTransformerEmbeddingFunction()
# print(embedding_function([token[0]]))

chromaCLient = chromadb.Client()
collection = chromaCLient.create_collection(name="last_lecture_collection", embedding_function=embedding_function)

ids = [str(i) for i in range(len(token))]
collection.add(ids=ids, documents=token)
# count = collection.count()
# print(f"Number of documents in the collection: {count}")

query = input("Enter your question: ")
results = collection.query(query_texts=[query], n_results=5)
retrieved_chunks = results['documents'][0]

def generate_augment_query(query, model="qwen3:0.6b"):
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

hypothetical_answer = generate_augment_query(query)
joint_query = f"{query} {hypothetical_answer}"

results = collection.query(query_texts=[joint_query], n_results=5, include=["documents", "embeddings"])
retrieved_document = results['documents'][0]

embeddings = collection.get(include=["embeddings"])['embeddings']
umap_transformer = umap.UMAP(n_components=0, transform_seed=0)
retrieved_embeddings = results['embeddings'][0]

# prepare arrays
embeddings_arr = np.array(embeddings)  # shape (n_docs, dim)
query_emb = np.array(embedding_function([query]))         # shape (1, dim)
joint_query_emb = np.array(embedding_function([joint_query]))  # shape (1, dim)

# fit UMAP (use n_components=2 for 2D)
umap_transformer = umap.UMAP(n_components=2, random_state=0)
projected_embeddings = umap_transformer.fit_transform(embeddings_arr)

# transform queries (transform requires a fitted transformer)
original_query_embedding = umap_transformer.transform(query_emb)
augmented_query_embedding = umap_transformer.transform(joint_query_emb)

plt.figure()
plt.scatter(projected_embeddings[:, 0], projected_embeddings[:, 1], label='Document Embeddings', s=10, color='gray')
plt.scatter(original_query_embedding[:, 0], original_query_embedding[:, 1], label='Original Query', s=100, color='red', marker='x')
plt.scatter(augmented_query_embedding[:, 0], augmented_query_embedding[:, 1], label='Augmented Query', s=100, color='green', marker='o')
plt.gca().set_aspect('equal', adjustable='datalim')
plt.title("UMAP Projection of Document and Query Embeddings")
plt.axis('off')
plt.show()
