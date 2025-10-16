import json
import os
import argparse

from .config import *
from .extracts import Extracts
from .chunks import Chunks
from .vectors import Vectors
from .questions import Questions
from .retrievers import Retrievers
from .evals import Evals
import logfire


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RAG Full Cycle Pipeline')
    parser.add_argument('dataset', nargs='?', default='fy10', 
                       choices=['fy10', 'baby'],
                       help='Dataset to process (fy10 or baby)')
    parser.add_argument('--steps', nargs='+', 
                       choices=['extract', 'chunk', 'vectorize', 'questions', 'retrievers', 'evaluate'],
                       help='Specific steps to run (default: all)')
    
    args = parser.parse_args()
    
    logfire.configure(token=LOGFIRE_API_KEY)
    logfire.info('Starting RAG Pipeline for dataset: {dataset}!', dataset=DATASET)
    logfire.info('PDF file: {pdf_file}', pdf_file=PDF_FILE_PATH)
    logfire.info('Chunk sizes: {sizes}, Overlaps: {overlaps}', sizes=CHUNK_SIZES, overlaps=CHUNK_OVERLAPS)

    extract_file = f"{OUTPUT_DIR}/extract.json"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extract step
    if not args.steps or 'extract' in args.steps:
        with logfire.span('extract PDF file'):
            if not os.path.exists(extract_file):
                extract_generator = Extracts()
                extract = extract_generator.extract_and_save(
                    PDF_FILE_PATH,
                    extract_file,
                    # max_pages=3
                )
            else:
                with open(extract_file, 'r') as f:
                    extract = json.load(f)
                logfire.info("Loaded {len} pages from existing extract file", len=len(extract))
    else:
        # Load existing extract if not running extract step
        with open(extract_file, 'r') as f:
            extract = json.load(f)
            logfire.info("Loaded {len} pages from existing extract file", len=len(extract))
    # Run pipeline for each configuration
    for size in CHUNK_SIZES:
        for overlap in CHUNK_OVERLAPS:
            logfire.info("Processing configuration: size={size}, overlap={overlap}", size=size, overlap=overlap)
            runPipelineForConfig(extract, size, overlap, args.steps)

    logfire.info('RAG Pipeline completed successfully!')

def runPipelineForConfig(extract, size, overlap, steps=None):
    """Run pipeline for a specific configuration with optional step filtering"""
    chunk_key = f"{size}-{overlap}"

    with logfire.span('processing {chunk_key}', chunk_key=chunk_key):
        # Output files for this configuration
        chunks_file = f"{OUTPUT_DIR}/chunks-{chunk_key}.json"
        vectors_file = f"{OUTPUT_DIR}/vectors-{chunk_key}.json"
        questions_file = f"{OUTPUT_DIR}/questions-{chunk_key}.json"
        retrievers_file = f"{OUTPUT_DIR}/retrievers-{chunk_key}.json"
        evals_file = f"{OUTPUT_DIR}/evals-{chunk_key}.json"

        # Chunking
        if not steps or 'chunk' in steps:
            with logfire.span('Chunking'):
                if not os.path.exists(chunks_file):
                    chunk_generator = Chunks(size, overlap)
                    chunks = chunk_generator.create_and_save_chunks(extract, chunks_file)
                else:
                    with open(chunks_file, 'r') as f:
                        chunks = json.load(f)
                    logfire.info('Found {chunks_file} chunk file', chunks_file=chunks_file)
        else:
            # Load existing chunks if not running chunk step
            with open(chunks_file, 'r') as f:
                chunks = json.load(f)
            logfire.info('Loaded existing chunks from {chunks_file}', chunks_file=chunks_file)

        # Early return if only running chunk step
        if steps and len(steps) == 1 and 'chunk' in steps:
            return

        # vectors
        if not steps or 'vectorize' in steps:
            with logfire.span('Vector creation'):
                if not os.path.exists(vectors_file):
                    vector_generator = Vectors(size, overlap)
                    success = vector_generator.process_chunks_to_vectors(chunks, vectors_file)
                    if not success:
                        logfire.info("Vector generation failed for")
                        return
                else:
                    logfire.info("Found {vectors_file} vector file", vectors_file=vectors_file)

        # questions
        if not steps or 'questions' in steps:
            with logfire.span('Questions generation'):
                if not os.path.exists(questions_file):
                    question_generator = Questions(size, overlap)
                    question_generator.generate_questions_from_chunks(chunks, questions_file)
                else:
                    logfire.info("Found {questions_file} questions file", questions_file=questions_file)

        # retrievers
        if not steps or 'retrievers' in steps:
            with logfire.span('Retrievers'):
                if not os.path.exists(retrievers_file):
                    retriever = Retrievers(size, overlap)
                    question_generator = Questions(size, overlap)
                    questions_data = question_generator.load_questions_from_file(questions_file)
                    
                    # 30%
                    total_questions = len(questions_data)
                    num_questions = max(10, min(total_questions, int(total_questions * 0.3)))
                    logfire.info("Using {num_questions} questions out of {total_questions} total (30% rule)", 
                               num_questions=num_questions, total_questions=total_questions)
                    
                    retrievers = retriever.run_tests_for_chunk_size(questions_data, retrievers_file, num_questions=num_questions)
                else:
                    with open(retrievers_file, 'r') as f:
                        retrievers = json.load(f)
                    logfire.info("Found {retrievers_file} retriever file", retrievers_file=retrievers_file)
        else:
            # Load existing retrievers if not running retrievers step
            with open(retrievers_file, 'r') as f:
                retrievers = json.load(f)
            logfire.info('Loaded existing retrievers from {retrievers_file}', retrievers_file=retrievers_file)

        # eval
        if not steps or 'evaluate' in steps:
            with logfire.span('Evals'):
                evaluator = Evals(questions_results=retrievers)
                evaluator.evaluate_and_save_results(k=3, evals_file=evals_file)

if __name__ == "__main__":
    main()
