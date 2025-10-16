from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .config import *
import requests
import logfire  

class Embeddings:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
   ## def create_embedding_openAI(self, text):
   ##     """Create embedding with automatic retry on rate limits"""
   ##     response = self.client.embeddings.create(
   ##         input=text,
   ##         model=EMBEDDING_MODEL_3_SMALL
   ##     )
   ##     return response.data[0].embedding
##
    def create_embedding_ollama(self, text):
        logfire.info("Creating embedding {text}", text=text)
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": text
            }
        )
        return response.json()["embeddings"][0]