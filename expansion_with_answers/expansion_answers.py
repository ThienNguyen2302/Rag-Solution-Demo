import os
from pypdf import PdfReader
from ollama import Client as OllamaClient 
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OLLAMA_KEY")
host = os.getenv("OLLAMA_HOST")

reader = PdfReader("data/fat_lost.pdf")
pdf_text = [page.extract_text().strip() for page in reader.pages]
texts = [text for text in pdf_text if text]

#split the text into smaller chunks