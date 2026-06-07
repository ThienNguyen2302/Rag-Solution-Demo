import os
import chromadb
from pypdf import PdfReader
from ollama import Client as OllamaClient 
from dotenv import load_dotenv
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv()
key = os.getenv("OLLAMA_KEY")
host = os.getenv("OLLAMA_HOST")

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
count = collection.count()
print(f"Number of documents in the collection: {count}")

query = input("Enter your question: ")
results = collection.query(query_texts=[query], n_results=5)
retrieved_chunks = results['documents'][0]

