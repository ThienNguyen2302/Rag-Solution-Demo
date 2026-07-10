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

question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

base_dir = Path(__file__).resolve().parent
pdf_path = base_dir / "data" / "last_lecture.pdf"
reader = PdfReader(str(pdf_path))
pdf_text = [page.extract_text().strip() for page in reader.pages]
texts = [text for text in pdf_text if text]

#split the text into smaller chunks
character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=400, chunk_overlap=50)
chunks = character_splitter.split_text("\n\n".join(texts))

# Encode the query
query = input("Enter your question: ")
query_encoding = question_tokenizer(query, return_tensors="pt")
query_embedding = question_encoder(**query_encoding).pooler_output

# Encode the chunks
context_embeddings = []
for chunk in chunks:
    context_encoding = context_tokenizer(chunk, return_tensors="pt")
    context_embedding = context_encoder(**context_encoding).pooler_output
    context_embeddings.append(context_embedding)

context_embeddings = torch.cat(context_embeddings, dim=0)

similarity_scores = cosine_similarity(query_embedding.detach().numpy(), context_embeddings.detach().numpy())

most_similar_indices = np.argsort(similarity_scores[0])[::-1][:5]
print("Top 5 most similar chunks:")
for idx in most_similar_indices:
    print(f"Chunk: {chunks[idx]}, Similarity Score: {similarity_scores[0][idx]:.4f}")





