import json
import os

from .config import *
from .extracts import Extracts
from .chunks import Chunks
from .vectors import Vectors
from .questions import Questions
from .retrievers import Retrievers
from .evals import Evals
import logfire


def runPipelineForConfig(extract, size, overlap, embedding_model):
    chunk_key = f"{size}-{overlap}"

    with logfire.span('processing {chunk_key} {embedding_model}', chunk_key=chunk_key, embedding_model=embedding_model):
        # Output files for this configuration
        chunks_file = f"{OUTPUT_DIR}/chunks-{chunk_key}.json"
        vectors_file = f"{OUTPUT_DIR}/vectors-{chunk_key}.json"
        questions_file = f"{OUTPUT_DIR}/questions-{chunk_key}.json"
        retrievers_file = f"{OUTPUT_DIR}/retrievers-{chunk_key}.json"

        # Chunking
        with logfire.span('Chinking'):
            if not os.path.exists(chunks_file):
                chunk_generator = Chunks(size, overlap)
                chunks = chunk_generator.create_and_save_chunks(extract, chunks_file)
            else:
                with open(chunks_file, 'r') as f:
                    chunks = json.load(f)
                logfire.info('Found {chunks_file} chunk file', chunks_file=chunks_file)

        # vectors
        with logfire.span('Vector creation'):
            if not os.path.exists(vectors_file):
                vector_generator = Vectors(size, overlap, embedding_model)
                success = vector_generator.process_chunks_to_vectors(chunks, vectors_file)
                if not success:
                    logfire.info("Vector generation failed for")
                    return
            else:
                logfire.info("Found {vectors_file} vector file", vectors_file=vectors_file)

        # questions
        with logfire.span('Questions generation'):
            if not os.path.exists(questions_file):
                question_generator = Questions(size, overlap, embedding_model)
                question_generator.generate_questions_from_chunks(chunks, questions_file)
            else:
                logfire.info("Found {questions_file} questions file", questions_file=questions_file)

        # retrievers
        with logfire.span('Retrievers'):
            if not os.path.exists(retrievers_file):
                retriever = Retrievers(size, overlap, embedding_model)
                retrievers = retriever.run_tests_for_chunk_size(questions_file, retrievers_file, num_questions=10)
            else:
                with open(retrievers_file, 'r') as f:
                    retrievers = json.load(f)
                logfire.info("Found {retrievers_file} retriever file", retrievers_file=retrievers_file)

        # eval (WIP) 
        with logfire.span('Evals'):
            evaluator = Evals(questions_results=retrievers)
            evaluator.evaluate_and_print_results(k=2)

def main():
    logfire.configure(token=LOGFIRE_API_KEY)
    logfire.info('Starting RAG Pipeline...!')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with logfire.span('extract PDF file'):
        if not os.path.exists(EXTRACT_OUTPUT):
            extract_generator = Extracts()
            extract = extract_generator.extract_and_save(
                PDF_FILE_PATH,
                EXTRACT_OUTPUT,
                # max_pages=3
            )
        else:
            with open(EXTRACT_OUTPUT, 'r') as f:
                extract = json.load(f)
            logfire.info("Loaded {len} pages from existing extract file", len=len(extract))

    for size in CHUNK_SIZES:
        for overlap in CHUNK_OVERLAPS:
            runPipelineForConfig(extract, size, overlap, EMBEDDING_MODEL_3_SMALL)

    logfire.info('RAG Pipeline completed successfully!')

if __name__ == "__main__":
    main()
