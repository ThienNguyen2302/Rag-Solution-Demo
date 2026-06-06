from pypdf import PdfReader
import os
from ollama import OllamaClient
from dotenv import load_dotenv

load_dotenv()
ollama = OllamaClient(host=os.getenv("OLLAMA_HOST"))