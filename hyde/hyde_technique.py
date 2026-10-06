from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from ollama import Client as OllamaClient
from dotenv import load_dotenv
import torch
import numpy as np
import os
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

load_dotenv()
key = os.getenv("OLLAMA_KEY")
host = os.getenv("OLLAMA_HOST")
ollama_client = OllamaClient(host=host)

# Initialize DPR models for encoding
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

base_dir = Path(__file__).resolve().parent
pdf_path = base_dir / "data" / "last_lecture.pdf"
reader = PdfReader(str(pdf_path))
pdf_text = [page.extract_text().strip() for page in reader.pages]
texts = [text for text in pdf_text if text]

# Split the text into smaller chunks
character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=400, chunk_overlap=50)
chunks = character_splitter.split_text("\n\n".join(texts))

# Handle case where no text was extracted
if not texts or not chunks:
    print("Error: No text could be extracted from the source file. Please check the file.")
    exit(1)

def generate_hypothetical_document(query, model="qwen3:0.6b"):
    """
    Generate a hypothetical document that answers the query.
    This is the core of HyDE - we create a fake document that
    would answer the user's question, then embed that to search.
    """
    prompt = f"""
    You are an expert AI assistant tasked with generating a hypothetical document
    that would answer the user's question based on the book "The Last Lecture" by Randy Pausch.

    Your task is to create a realistic passage that sounds like it could be extracted
    directly from the book, answering the user's question as comprehensively as possible.

    ### Instructions:
    1. Analyze the user's question to understand what information they're seeking
    2. Generate a hypothetical document passage that answers this question
    3. Write in the style and tone of Randy Pausch from "The Last Lecture"
    4. Use specific terminology, concepts, and examples from the book where relevant
    5. Make it sound authentic - like it could be a real passage from the book
    6. The hypothetical document should be detailed enough to be useful for retrieval

    ### Output Format:
    Provide ONLY the hypothetical document text as a single paragraph.
    Do not include any explanations, metadata, or formatting.

    User's question: {query}
    """

    messages = [
        {"role": "system", "content": prompt.strip()},
        {"role": "user", "content": "Generate the hypothetical document now."}
    ]

    response = ollama_client.chat(model=model, messages=messages)
    return response['message']['content'].strip()

def encode_text(text, encoder, tokenizer):
    """Encode text using the provided encoder and tokenizer."""
    encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        embedding = encoder(**encoding).pooler_output
    return embedding

# Encode all chunks once for efficiency
print("Encoding document chunks...")
context_embeddings = []

for i, chunk in enumerate(chunks):
    if i % 50 == 0:
        print(f"  Processed {i}/{len(chunks)} chunks")
    context_embedding = encode_text(chunk, context_encoder, context_tokenizer)
    context_embeddings.append(context_embedding)

context_embeddings = torch.cat(context_embeddings, dim=0)
print(f"Encoded {len(context_embeddings)} document chunks")

# Main HyDE process
print("\n" + "="*50)
print("HYDE (Hypothetical Document Embedding) TECHNIQUE")
print("="*50)

query = input("\nEnter your question: ")

# Step 1: Generate hypothetical document
print("\nGenerating hypothetical document...")
hypothetical_doc = generate_hypothetical_document(query)
print(f"Hypothetical document generated ({len(hypothetical_doc)} chars)")

# Step 2: Encode the hypothetical document
print("Encoding hypothetical document...")
hypothetical_embedding = encode_text(hypothetical_doc, context_encoder, context_tokenizer)

# Step 3: Search for real documents similar to the hypothetical document
print("Searching for similar real documents...")
similarity_scores = cosine_similarity(
    hypothetical_embedding.detach().numpy(),
    context_embeddings.detach().numpy()
)

most_similar_indices = np.argsort(similarity_scores[0])[::-1][:5]
print("\nTop 5 most similar chunks:")

retrieved_chunks = []
for i, idx in enumerate(most_similar_indices, 1):
    score = similarity_scores[0][idx]
    chunk_text = chunks[idx]
    retrieved_chunks.append(chunk_text)
    print(f"{i}. Similarity Score: {score:.4f}")
    print(f"   Chunk: {chunk_text[:150]}..." if len(chunk_text) > 150 else f"   Chunk: {chunk_text}")
    print()

# Step 4: Generate final answer using retrieved chunks
def generate_answer(query, retrieved_chunks, model="qwen3:0.6b"):
    prompt = f"""
    You are an expert AI assistant providing answers based on retrieved information
    from the book "The Last Lecture" by Randy Pausch.

    Based on the retrieved chunks of text from the book, provide a concise and accurate
    answer to the user's question. Use only the information contained in the retrieved
    chunks to formulate your response. Do not include any information that is not present
    in the retrieved chunks.

    ### Retrieved Chunks:
    {chr(10).join([f"- {chunk}" for chunk in retrieved_chunks])}

    ### Instructions:
    1. Analyze the retrieved chunks to find relevant information that directly answers the user's question
    2. Synthesize the information into a clear and concise answer
    3. Do NOT include any personal opinions or information that is not supported by the retrieved chunks

    ### Output Format:
    Provide your answer as a single paragraph of text without any markdown formatting or additional explanations.
    """.strip()

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query}
    ]

    response = ollama_client.chat(model=model, messages=messages)
    return response['message']['content']

print("\nGenerating final answer...")
final_answer = generate_answer(query, retrieved_chunks)
print("="*50)
print("FINAL ANSWER:")
print("="*50)
print(final_answer)
print("="*50)

# Optional: Show comparison with direct query encoding
print("\n" + "="*50)
print("COMPARISON WITH DIRECT QUERY ENCODING")
print("="*50)

# Encode the original query for comparison
query_encoding = question_tokenizer(query, return_tensors="pt")
query_embedding = question_encoder(**query_encoding).pooler_output

query_similarity_scores = cosine_similarity(
    query_embedding.detach().numpy(),
    context_embeddings.detach().numpy()
)

query_most_similar_indices = np.argsort(query_similarity_scores[0])[::-1][:5]
print("Top 5 chunks using direct query encoding:")

for i, idx in enumerate(query_most_similar_indices, 1):
    score = query_similarity_scores[0][idx]
    chunk_text = chunks[idx]
    print(f"{i}. Similarity Score: {score:.4f}")
    print(f"   Chunk: {chunk_text[:100]}..." if len(chunk_text) > 100 else f"   Chunk: {chunk_text}")
    print()