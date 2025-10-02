import json
from openai import OpenAI
from pinecone import Pinecone
import instructor
from .config import *
from .models import QuestionsResponse


class QuestionGenerator:
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
            prompt = QUESTION_GENERATION_PROMPT.format(chunk=chunk_text, grade_level=GRADE_LEVEL)
            
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_model=QuestionsResponse,
            )
            
            return response
        except Exception as e:
            print(f"Error generating questions for chunk: {e}")
            # Return empty questions response
            return QuestionsResponse(questions=[])
    
    def process_chunks(self, chunks):
        """Process chunks and generate questions"""
        print(f"Processing {len(chunks)} chunks...")

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
        print("\nGenerating questions from chunks...")
        
        try:
            # Generate questions for all chunks
            questions_data = self.process_chunks(chunks)
            
            # Save questions data
            self.save_questions(questions_data, output_path)
            
            return questions_data
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []
    
    def save_questions(self, questions_data, output_path):
        """Save questions data to file"""
        print(f"Saving questions to {output_path}")
        with open(output_path, "w") as f:
            json.dump(questions_data, f, indent=2)
        print(f"Saved {len(questions_data)} question sets to {output_path}")
