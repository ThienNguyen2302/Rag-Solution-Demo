from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)
from pathlib import Path
from pypdf import PdfReader
from ollama import Client as OllamaClient 
from dotenv import load_dotenv
import torch
import numpy as np
import os
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from turbovec import TurboQuantIndex

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

# Encode the chunks
context_embeddings = []
for chunk in chunks:
    context_encoding = context_tokenizer(chunk, return_tensors="pt")
    with torch.no_grad():
        context_embedding = context_encoder(**context_encoding).pooler_output
    context_embeddings.append(context_embedding)

context_embeddings = torch.cat(context_embeddings, dim=0)

# DPR of facebook has a dimension of 768, so we will use that for the TurboQuantIndex
kv_store = TurboQuantIndex(dim=768, bit_width=4)

#store the context embeddings in the kv_store
embeddings_numpy = context_embeddings.numpy()

kv_store.add(embeddings_numpy)

# for idx, vector in enumerate(embeddings_numpy):    
#     vector_list = vector.tolist()
#     kv_store._store_texts_and_vectors(key=idx, vector=vector_list, val=chunks[idx])

# # Encode the query
query = input("Enter your question: ")
query_encoding = question_tokenizer(query, return_tensors="pt")

# # with torch.no_grad() is used to disable gradient calculation
# # which is used for training, but since we are doing inference, we don't need gradients
# # it reduces memory consumption and speeds up computations during inference. 
with torch.no_grad():
    query_embedding = question_encoder(**query_encoding).pooler_output

query_vector = query_embedding[0].numpy().reshape(1, -1).astype(np.float32)
search_results = kv_store.search(query_vector, k=2)

retrieved_contexts = []

for score, idx in zip(scores[0], indices[0]):
    text_content = chunks[idx]
    retrieved_contexts.append(text_content)


