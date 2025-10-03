import json
from openai import OpenAI
from pinecone import Pinecone
import instructor
import logfire
from .config import *
from .models import QuestionsResponse

class Questions:
    def __init__(self, size, overlap, embedding_model):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        self.client = instructor.patch(self.client)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.embedding_model = embedding_model
        self.pinecone_index_name = f"{size}-{overlap}-{embedding_model}"
    
    def generate_questions_for_chunk(self, chunk_text):
        """Generate questions for a single chunk of text using Instructor"""
        try:
            logfire.info("Generation questions for: {chunk_text}", chunk_text=chunk_text)
            prompt = QUESTION_GENERATION_PROMPT.format(chunk=chunk_text, grade_level=GRADE_LEVEL)
            
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_model=QuestionsResponse,
            )
            
            return response
        except Exception as e:
            logfire.error("Error generating questions for chunk: {e}", e=e)
            # Return empty questions response
            return QuestionsResponse(questions=[])
    
    def process_chunks(self, chunks):
        """Process chunks and generate questions"""
        logfire.info("Processing {len} chunks...", len=len(chunks))

        questions_data = []
        for chunk in chunks:
            chunk_text = chunk['text']
            questions_response = self.generate_questions_for_chunk(chunk_text)

            questions_data.append({
                "chunk_id": chunk['id'],
                "text": chunk_text,
                "questions": questions_response.questions
            })
        return questions_data

    
    def generate_questions_from_chunks(self, chunks, output_path):
        """Generate questions from chunks"""
        logfire.info("Generating questions from chunks...")
        
        try:
            questions_data = self.process_chunks(chunks)
            self.save_questions(questions_data, output_path)
            return questions_data
            
        except Exception as e:
            logfire.error("Error generating questions: {e}", e=e)
            return []
    
    def save_questions(self, questions_data, output_path):
        """Save questions data to file"""
        logfire.info("Saving questions to {output_path}", output_path=output_path)
        with open(output_path, "w") as f:
            json.dump(questions_data, f, indent=2)
        logfire.info("Saved {len} question sets to {output_path}", len=len(questions_data), output_path=output_path)
