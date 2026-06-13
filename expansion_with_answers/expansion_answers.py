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

def generate_answer_expansion(query, model="qwen3:0.6b"):
    prompt = """
        You are an expert AI search assistant optimizing answers for a Retrieval-Augmented Generation (RAG) system using the answer expansion technique. 
        Your specific domain of expertise is the book "The Last Lecture" by Randy Pausch.

        Your task is to take a raw user query and generate a list of 3 to 5 hypothetical answers or textbook-style text chunks. These generated answers do not need to be 100% historically accurate, but they MUST mimic the exact tone, specific terminology, and narrative style of Randy Pausch in "The Last Lecture". These outputs will be embedded to search a vector database for semantic matching with the actual book chunks.

        ### Instructions:
        1. Analyze the user's core intent and identify the underlying theme from the book.
        2. Write 3 to 5 distinct, hypothetical paragraphs or statements that answer the query. 
        3. Use specific book terminology, concepts, and names where appropriate (e.g., brick walls, childhood dreams, Tigger vs. Eeyore, head fakes, the First Penguin Award, Jai, Chloe, Dylan, Logan).
        4. Write the outputs in the first-person perspective ("I", "my") or direct advice style, just like Randy Pausch delivered his lecture.
        5. Do NOT write questions. ONLY output hypothetical answer text chunks that look like they were ripped directly from the book pages.

        ### Output Format:
        You must respond ONLY with a valid JSON array of strings. Do not include markdown formatting, introductory text, or explanations.

        Example Input: "what did he say about failure?"
        Example Output: 
        [
        "Experience is what you get when you didn't get what you wanted. At Carnegie Mellon, I even gave out the First Penguin Award to the team that took the biggest financial or technical risk and failed, because learning from failure is essential.",
        "The brick walls are there for a reason. The brick walls are not there to keep us out. The brick walls are there to give us a chance to show how badly we want something. They are there to stop the people who don't want it badly enough.",
        "Failure is not just acceptable, it's an essential tool for growth. When you are doing something hard, you are going to screw up. The key is to fail fast, fail well, and treat mistakes as valuable lessons for the next attempt."
        ]
    """.strip()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query}
    ]

    response = ollama_client.chat(model=model, messages=messages)
    return response['message']['content']

answer_expansion = generate_answer_expansion(query)
expanded_query = f"{query} {answer_expansion}"

results = collection.query(query_texts=[expanded_query], n_results=5, include=["documents", "embeddings"])
retrieved_document = results['documents'][0]

embeddings = collection.get(include=["embeddings"])['embeddings']
umap_transformer = umap.UMAP(n_components=0, transform_seed=0)
retrieved_embeddings = results['embeddings'][0]

# prepare arrays
embeddings_arr = np.array(embeddings)  # shape (n_docs, dim)
query_emb = np.array(embedding_function([query]))         # shape (1, dim)
expanded_query_emb = np.array(embedding_function([expanded_query]))  # shape (1, dim)

# fit UMAP (use n_components=2 for 2D)
umap_transformer = umap.UMAP(n_components=2, random_state=0)
projected_embeddings = umap_transformer.fit_transform(embeddings_arr)

# transform queries (transform requires a fitted transformer)
original_query_embedding = umap_transformer.transform(query_emb)
expanded_query_embedding = umap_transformer.transform(expanded_query_emb)

plt.figure()
plt.scatter(projected_embeddings[:, 0], projected_embeddings[:, 1], label='Document Embeddings', s=10, color='gray')
plt.scatter(original_query_embedding[:, 0], original_query_embedding[:, 1], label='Original Query', s=100, color='red', marker='x')
plt.scatter(expanded_query_embedding[:, 0], expanded_query_embedding[:, 1], label='Expanded Answer', s=100, color='green', marker='o')
plt.gca().set_aspect('equal', adjustable='datalim')
plt.title("UMAP Projection of Document and Query Embeddings")
plt.axis('off')
plt.show()

def generate_answer(query, retrieved_chunks, model="qwen3:0.6b"):
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

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query}
    ]

    response = ollama_client.chat(model=model, messages=messages)
    return response['message']['content']

final_answer = generate_answer(query, retrieved_document)
print(f"Final Answer: {final_answer}")