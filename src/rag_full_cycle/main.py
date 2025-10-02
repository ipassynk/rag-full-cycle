import json
import os
import glob
import random

from .config import *
from .extract_generator import ExtractGenerator
from .chunk_generator import ChunkGenerator
from .vector_generator import VectorGenerator
from .question_generator import QuestionGenerator
from .embedding_generator import EmbeddingGenerator
from .question_tester import RandomQuestionTester

def main():
    print("Starting RAG Pipeline...")
    print("=" * 50)
    
    extract = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Extract text 
    if not os.path.exists(EXTRACT_OUTPUT):
        extract_generator = ExtractGenerator()
        extract = extract_generator.extract_and_save(
            PDF_FILE_PATH, 
            EXTRACT_OUTPUT, 
            #max_pages=3
        )
    else:
        with open(EXTRACT_OUTPUT, 'r') as f:
            extract = json.load(f)
        print(f"Loaded {len(extract)} pages from existing extract file")

    # Generate chunks
    existing_chunk_files = glob.glob(f"{OUTPUT_DIR}/chunks-*.json")    
    if not existing_chunk_files:
        chunk_generator = ChunkGenerator()
        all_chunks = chunk_generator.process_all_chunk_sizes(extract)
    else:
        all_chunks = {}
        for chunk_file in existing_chunk_files:
            # Extract chunk key from filename (e.g., "chunks-100-10.json" -> "100-10")
            chunk_key = os.path.basename(chunk_file).replace("chunks-", "").replace(".json", "")
            with open(chunk_file, 'r') as f:
                all_chunks[chunk_key] = json.load(f)
        print(f"Loaded {len(all_chunks)} chunk files")

    # Generate vectors
    existing_vector_files = glob.glob(f"{OUTPUT_DIR}/vectors-*.json")
    if not existing_vector_files:
        for chunk_key, chunks in all_chunks.items():
            print(f"\nProcessing chunk size: {chunk_key}")
            size, overlap = chunk_key.split("-")
            vector_generator = VectorGenerator(size, overlap, embedding_model=EMBEDDING_MODEL_3_SMALL)
            vectors_output = f"{OUTPUT_DIR}/vectors-{chunk_key}.json"
            success = vector_generator.process_chunks_to_vectors(chunks, vectors_output)
            
            if not success:
                print(f"Vector generation failed for {chunk_key}, continuing with next chunk size")
                continue
    else:
        print(f"Found {len(existing_vector_files)} vector files")

    # Generate questions
    existing_questions_files = glob.glob(f"{OUTPUT_DIR}/questions-*.json")
    if not existing_questions_files:
        print("Question files not found, generating questions...")
        for chunk_key, chunks in all_chunks.items():
            size, overlap = chunk_key.split("-")
            question_generator = QuestionGenerator(size, overlap, embedding_model=EMBEDDING_MODEL_3_SMALL)
            questions_output = f"{OUTPUT_DIR}/questions-{chunk_key}.json"
            question_generator.generate_questions_from_chunks(chunks, questions_output)
    else:
        print(f"Found {len(existing_questions_files)} question files")

    # Test random questions for all chunk sizes
    all_results = {}
    for chunk_key, chunks in all_chunks.items():
        print(f"\nTesting chunk size: {chunk_key}")
        size, overlap = chunk_key.split("-")
        
        tester = RandomQuestionTester(size, overlap, embedding_model=EMBEDDING_MODEL_3_SMALL)
        output_path = f"{OUTPUT_DIR}/tester-{chunk_key}.json"
        results = tester.run_tests_for_chunk_size(chunk_key, output_path, num_questions=10)
        all_results[chunk_key] = results

    print("\nRAG Pipeline completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
