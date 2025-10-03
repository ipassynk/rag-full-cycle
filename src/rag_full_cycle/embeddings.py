from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .config import *

class Embeddings:
    def __init__(self, size, overlap, embedding_model):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.embedding_model = embedding_model
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def create_embedding_with_retry(self, text):
        """Create embedding with automatic retry on rate limits"""
        return self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
    