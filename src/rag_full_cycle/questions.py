import json
import re
from pinecone import Pinecone
import logfire
from .config import *
from ollama import chat
from ollama import ChatResponse
from concurrent.futures import ThreadPoolExecutor, as_completed

class Questions:
    def __init__(self, size, overlap):
        #self.client = OpenAI(
        #    base_url="https://openrouter.ai/api/v1",
        #    api_key=OPENROUTER_API_KEY,
        #)
        #self.client = instructor.patch(self.client)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.pinecone_index_name = f"{size}-{overlap}-{pdf_file}"

    def load_questions_from_file(self, questions_file):
        with open(questions_file, 'r') as f:
             questions = json.load(f)
        return questions

    def generate_questions_for_chunk(self, chunk_text):
        """Generate questions for a single chunk of text using Instructor"""
        try:
            logfire.info("Generation questions for: {chunk_text}", chunk_text=chunk_text)
            prompt = QUESTION_GENERATION_PROMPT.format(chunk=chunk_text)
            # too much 429 
            #response = self.client.chat.completions.create(
            #    model=OPEN_ROUTER_MODEL,
            #    messages=[{"role": "user", "content": prompt}]
            #)
            #content = response.choices[0].message.content
            response: ChatResponse = chat(model=QUESTION_MODEL, 
                messages=[{"role": "user", "content": prompt}]
            )
            content = response['message']['content'];
            questions = [q.strip() for q in content.split('\n') if q.strip()]
            # in case the questions are numbered
            clean_questions = [re.sub(r'^\s*\d+\.\s*', '', q) for q in questions]

            return clean_questions
        except Exception as e:
            logfire.error("Error generating questions for chunk: {e}", e=e)
            raise e
    
    def process_chunks(self, chunks):
        logfire.info("Processing {len} chunks...", len=len(chunks))

        questions_data = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_chunk = {}
            for chunk in chunks:
                future = executor.submit(self.process_single_chunk, chunk)
                future_to_chunk[future] = chunk
            
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    questions_data.append(result)
                    logfire.info("Completed processing chunk {id}", id=chunk['id'])
                except Exception as e:
                    logfire.error("Error processing chunk {id}: {e}", id=chunk['id'], e=e)
        
        questions_data.sort(key=lambda x: [int(part) for part in x['chunk_id'].split('-')])
        return questions_data
    
    def process_single_chunk(self, chunk):
        chunk_text = chunk['text']
        logfire.info("Generation questions for chunk {id}...", id=chunk['id'])
        
        questions = self.generate_questions_for_chunk(chunk_text)
        return {
            "chunk_id": chunk['id'],
            "text": chunk_text,
            "questions": questions
        }

    
    def generate_questions_from_chunks(self, chunks, output_path):
        """Generate questions from chunks"""
        logfire.info("Generating questions from chunks...")
        
        try:
            questions_data = self.process_chunks(chunks)
            self.save_questions(questions_data, output_path)
            return questions_data
            
        except Exception as e:
            logfire.error("Error generating questions: {e}", e=e)
            raise e
    
    def save_questions(self, questions_data, output_path):
        """Save questions data to file"""
        logfire.info("Saving questions to {output_path}", output_path=output_path)
        with open(output_path, "w") as f:
            json.dump(questions_data, f, indent=2)
        logfire.info("Saved {len} question sets to {output_path}", len=len(questions_data), output_path=output_path)
