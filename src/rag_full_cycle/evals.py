from typing import List, Dict, Any
import logfire  

class Evals:
    """Evaluate RAG retrieval results with Recall@K metrics"""
    
    def __init__(self, questions_results: List[Dict[str, Any]]):
        self.questions_results = questions_results
    
    def calculate_recall_at_k(self, k) -> float:
        """
        Calculate Recall@K for the given results
        """
        correct_retrievals = 0
        total_queries = len(self.questions_results)
        
        for result in self.questions_results:
            question_data = result['question_data']
            similar_chunks = result['similar_chunks']
            
            # Get the correct chunk ID (the one the question was generated from)
            correct_chunk_id = question_data['chunk_id']
            
            # Check if correct chunk is in top-K results
            top_k_chunks = similar_chunks[:k]
            chunk_ids = [chunk['id'] for chunk in top_k_chunks]
            
            if correct_chunk_id in chunk_ids:
                correct_retrievals += 1
        
        recall_at_k = correct_retrievals / total_queries if total_queries > 0 else 0.0
        return recall_at_k

    
    def evaluate_and_print_results(self,k):
        """
        Evaluate results and print metrics
        """
        recall_result = self.calculate_recall_at_k(k)
        logfire.info("\nRecall@{k}: {recall_result:.2%}\n", k=k, recall_result=recall_result)
    
