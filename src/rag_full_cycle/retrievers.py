import json
import random
import os
import logfire
from .config import *
from .embeddings import Embeddings
from .vectors import Vectors


class Retrievers:
    """Handles random question selection and testing for RAG pipeline"""
    
    def __init__(self, size, overlap, embedding_model=EMBEDDING_MODEL_3_SMALL):
        self.size = size
        self.overlap = overlap
        self.embedding_model = embedding_model
        self.embedding_generator = Embeddings(size, overlap, embedding_model)
        self.vector_generator = Vectors(size, overlap, embedding_model)
    
    def load_questions_from_file(self, questions_file):
        with open(questions_file, 'r') as f:
            questions = json.load(f)
        return questions
    
    def extract_all_questions(self, questions_data):
        all_questions = []
        for question_data in questions_data:
            chunk_id = question_data['chunk_id']
            text = question_data['text']
            questions_list = question_data['questions']
            
            # Add each question with its context
            for question in questions_list:
                all_questions.append({
                    'chunk_id': chunk_id,
                    'text': text,
                    'question': question
                })
        
        return all_questions
    
    def select_random_questions(self, all_questions, num_questions=10):
        if len(all_questions) >= num_questions:
            selected_questions = random.sample(all_questions, num_questions)
        else:
            selected_questions = all_questions
        
        logfire.info("Selected {len} questions for testing", len=len(selected_questions))
        return selected_questions
    
    def test_question_retrieval(self, question_data):
        logfire.info("Test question: {question_data}", question_data=question_data)
        
        try:
            embedding_response = self.embedding_generator.create_embedding_with_retry(question_data['question'])
            question_embedding = embedding_response.data[0].embedding
            
            similar_chunks = self.vector_generator.find_similar_chunks(question_embedding)
            return similar_chunks
            
        except Exception as e:
            logfire.error("Error testing question retrieval: {e}", e=e)
            return []
    
    def run_question_tests(self, questions_file, num_questions=10):
        logfire.info("Running question tests for {questions_file}", questions_file=questions_file)

        questions_data = self.load_questions_from_file(questions_file)
        all_questions = self.extract_all_questions(questions_data)
        selected_questions = self.select_random_questions(all_questions, num_questions)
        
        results = []
        for question_data in selected_questions:
            similar_chunks = self.test_question_retrieval(question_data)
            results.append({
                'question_data': question_data,
                'similar_chunks': similar_chunks
            })
        
        return results
    
    def run_tests_for_chunk_size(self, questions_file, output_path, num_questions=10):
        results = self.run_question_tests(questions_file, num_questions)
        self.save_questions(results, output_path)
        return results
    
    def save_questions(self, questions_data, output_path):
        logfire.info("Saving questions to {output_path}", output_path=output_path)
        with open(output_path, "w") as f:
            json.dump(questions_data, f, indent=2)
        logfire.info("Saved {len} questions to {output_path}", len=len(questions_data), output_path=output_path)