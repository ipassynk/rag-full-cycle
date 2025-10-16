from typing import List, Dict, Any
import logfire
import json

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
            
            correct_chunk_id = question_data['chunk_id']
            
            # Check if correct chunk is in top-K results
            top_k_chunks = similar_chunks[:k]
            chunk_ids = [chunk['id'] for chunk in top_k_chunks]
            
            if correct_chunk_id in chunk_ids:
                correct_retrievals += 1
        
        recall_at_k = correct_retrievals / total_queries if total_queries > 0 else 0.0
        return recall_at_k

    def calculate_precision_at_k(self, k) -> float:
        """
        Precision@K = (Number of relevant documents in top-K) / K
        """
        total_precision = 0.0
        total_queries = len(self.questions_results)
        
        for result in self.questions_results:
            question_data = result['question_data']
            similar_chunks = result['similar_chunks']
            
            correct_chunk_id = question_data['chunk_id']
            
            top_k_chunks = similar_chunks[:k]
            chunk_ids = [chunk['id'] for chunk in top_k_chunks]
            
            # Count relevant documents in top-K (in this case, 1 if correct chunk is found)
            relevant_in_top_k = 1 if correct_chunk_id in chunk_ids else 0
            
            # Precision for this query = relevant documents / K
            precision_for_query = relevant_in_top_k / k
            total_precision += precision_for_query
        
        avg_precision_at_k = total_precision / total_queries if total_queries > 0 else 0.0
        return avg_precision_at_k

    
    def evaluate_and_save_results(self,k,evals_file):
        """
        Evaluate results and print metrics
        """
        recall_result = self.calculate_recall_at_k(k)
        logfire.info("\nRecall@{k}: {recall_result:.2%}\n", k=k, recall_result=recall_result)
        precision_result = self.calculate_precision_at_k(k)
        logfire.info("Precision@{k}: {precision_result:.2%}\n", k=k, precision_result=precision_result)
        evals_data = {
            "recall": recall_result,
            "precision": precision_result
        }
        self.save_evals(evals_data, evals_file)
        
    def save_evals(self, evals_data, output_path):
        logfire.info("Saving evals to {output_path}", output_path=output_path)
        with open(output_path, "w") as f:
            json.dump(evals_data, f, indent=2)