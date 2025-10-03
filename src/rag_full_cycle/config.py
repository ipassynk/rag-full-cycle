import os
from typing import List

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
LOGFIRE_API_KEY = os.environ.get("LOGFIRE_API_KEY")

# File 
PDF_FILE_PATH = "fy10syb.pdf"
PDF_FILE_PATH = "The-Three-Little-Pigs-original.pdf"

OUTPUT_DIR = "output"
EXTRACT_OUTPUT = f"{OUTPUT_DIR}/extract.json"
CHUNKS_OUTPUT = f"{OUTPUT_DIR}/chunks.json"

# Chunking Configuration
#CHUNK_SIZES: List[int] = [100, 256, 512]
#CHUNK_OVERLAPS: List[int] = [10, 64, 64]

### This is in words
CHUNK_SIZES: List[int] = [100]
CHUNK_OVERLAPS: List[int] = [10]

# OpenAI Configuration
OPENAI_MODEL = "x-ai/grok-4-fast:free"
EMBEDDING_MODEL_3_SMALL = "text-embedding-3-small"
EMBEDDING_MODEL_3_LARGE = "text-embedding-3-large"

# Vector Search Configuration
TOP_K_RESULTS = 10
PINECONE_NAMESPACE = "default"

# Rate Limiting Configuration
BATCH_SIZE = 10  # Process chunks in batches
DELAY_BETWEEN_REQUESTS = 0.1  # 100ms delay between requests
DELAY_BETWEEN_BATCHES = 1.0  # 1 second delay between batches
MAX_RETRIES = 3  # Maximum retry attempts for rate limits

# Prompt Configuration
GRADE_LEVEL = 2
QUESTION_GENERATION_PROMPT = """
You are an educational expert creating questions for a classroom setting.

Your task is to generate 3 questions for the given text chunk that simulate realistic teacher and student interactions:
This is the grade level: {grade_level}

Question Types:
1. TEACHER QUESTION: A direct, clear question that a teacher would ask to test comprehension
2. STUDENT QUESTION: A curious question that a student might ask when learning this topic
3. ADVANCED QUESTION: A deeper, analytical question that challenges understanding

Guidelines:
- Use natural, conversational language
- Make questions age-appropriate and engaging
- Ensure questions can be answered from the given text
- Vary the complexity and perspective
- Make them sound like real classroom interactions
- Return ONLY the question text, no labels or numbering

Text chunk:
\"\"\"
{chunk}
\"\"\"

Generate exactly 3 questions following the specified types and guidelines. 
Return only the question text without any labels, numbering, or prefixes.
"""
