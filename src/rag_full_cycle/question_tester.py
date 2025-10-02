import json
import random
import os
from .config import *
from .embedding_generator import EmbeddingGenerator
from .vector_generator import VectorGenerator


class RandomQuestionTester:
    """Handles random question selection and testing for RAG pipeline"""
    
    def __init__(self, size, overlap, embedding_model=EMBEDDING_MODEL_3_SMALL):
        self.size = size
        self.overlap = overlap
        self.embedding_model = embedding_model
        self.embedding_generator = EmbeddingGenerator(size, overlap, embedding_model)
        self.vector_generator = VectorGenerator(size, overlap, embedding_model)
    
    def load_questions_from_file(self, question_file):
        """Load questions from a JSON file"""
        with open(question_file, 'r') as f:
            questions = json.load(f)
        return questions
    
    def extract_all_questions(self, questions_data):
        """Extract all questions from the questions data structure"""
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
        """Select a random sample of questions for testing"""
        if len(all_questions) >= num_questions:
            selected_questions = random.sample(all_questions, num_questions)
        else:
            selected_questions = all_questions
        
        print(f"Selected {len(selected_questions)} questions for testing")
        return selected_questions
    
    def test_question_retrieval(self, question_data):
        """Test question retrieval by finding similar chunks"""
        print(f"Test question: {question_data['question']}")
        
        try:
            embedding_response = self.embedding_generator.create_embedding_with_retry(question_data['question'])
            question_embedding = embedding_response.data[0].embedding
            
            similar_chunks = self.vector_generator.find_similar_chunks(question_embedding)
            return similar_chunks
            
        except Exception as e:
            print(f"Error testing question retrieval: {e}")
            return []
    
    def run_question_tests(self, question_file, num_questions=10):
        """Run random question tests for a given question file"""
        print(f"\nRunning question tests for {question_file}")
        print("=" * 50)
        
        questions_data = self.load_questions_from_file(question_file)
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
    
    def run_tests_for_chunk_size(self, chunk_key, output_path, num_questions=10):
        """Run tests for a specific chunk size"""
        question_file = f"{OUTPUT_DIR}/questions-{chunk_key}.json"
        
        if not os.path.exists(question_file):
            print(f"Question file not found: {question_file}")
            return []
        
        results = self.run_question_tests(question_file, num_questions)
        self.save_questions(results, output_path)
        return results
    
    def save_questions(self, questions_data, output_path):
        """Save questions to file"""
        print(f"Saving questions to {output_path}")
        with open(output_path, "w") as f:
            json.dump(questions_data, f, indent=2)
        print(f"Saved {len(questions_data)} questions to {output_path}")