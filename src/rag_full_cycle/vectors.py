import json
import time
from openai import OpenAI
import logfire
from pinecone import Pinecone, ServerlessSpec
from .config import *
from .embeddings import Embeddings

class Vectors:
    def __init__(self, size, overlap):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.size = size
        self.overlap = overlap
        self.pinecone_index_name = f"{size}-{overlap}-{FILE_NAME}"
    
    def generate_embeddings_for_chunks(self, chunks):
        """Generate embeddings for a list of chunks"""
        logfire.info("Generating embeddings for {len} chunks...", len=len(chunks))
        logfire.info("Using batch size: {BATCH_SIZE}, delay: {DELAY_BETWEEN_REQUESTS}s", BATCH_SIZE=BATCH_SIZE, DELAY_BETWEEN_REQUESTS=DELAY_BETWEEN_REQUESTS)
        
        embedding_generator = Embeddings()
        vectors_to_upsert = []
        
        # Process chunks in batches to improve performance
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(chunks))
            batch = chunks[batch_start:batch_end]
            
            logfire.info("Processing batch {batch_start}: chunks {batch_end}-{BATCH_SIZE}", batch_start=batch_start, batch_end=batch_end, BATCH_SIZE=BATCH_SIZE)
            
            for i, chunk in enumerate(batch):
                try:
                    chunk_index = batch_start + i
                    logfire.info("Processing chunk {chunk_index}: {chunk_id}", chunk_index=chunk_index,  chunk_id=chunk['id'])
                    
                    # Create embedding
                    embedding = embedding_generator.create_embedding_ollama(chunk["text"])
                    vectors_to_upsert.append({
                        "id": chunk["id"], 
                        "values": embedding
                    })
                    
                    time.sleep(DELAY_BETWEEN_REQUESTS)
                    
                except Exception as e:
                    logfire.error("Error processing chunk {chunk_id}: {e}", chunk_id=chunk['id'], e=e)
                    continue
            
            if batch_end < len(chunks):
                logfire.info("Waiting {DELAY_BETWEEN_BATCHES}s before next batch...", DELAY_BETWEEN_BATCHES=DELAY_BETWEEN_BATCHES)
                time.sleep(DELAY_BETWEEN_BATCHES)
        
        return vectors_to_upsert
    
    def save_vectors(self, vectors, output_path):
        """Save vectors to file"""
        logfire.info("Saving {len} vectors to {output_path}", len=len(vectors), output_path=output_path)
        with open(output_path, "w") as f:
            json.dump(vectors, f, indent=2)
        logfire.info("Saved vectors to {output_path}", output_path=output_path)
    
    def create_and_manage_index(self, vectors, dimension):
        """Create Pinecone index if needed and return index object"""
        logfire.info("Index {pinecone_index_name} creating one...", pinecone_index_name=self.pinecone_index_name)
        self.pc.create_index(
            name=self.pinecone_index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        time.sleep(10)
        # Get the index object
        index = self.pc.Index(self.pinecone_index_name)
        return index
    
    def upsert_vectors_to_pinecone(self, vectors, dimension):
        """Upsert vectors to Pinecone"""
        try:
            logfire.info("Upserting {len} vectors to Pinecone, dimension: {dimension}...", len=len(vectors), dimension=dimension)
            
            index = self.create_and_manage_index(vectors, dimension)
            index.upsert(
                namespace=PINECONE_NAMESPACE,
                vectors=vectors
            )
            logfire.info("Successfully upserted {len} vectors to Pinecone", len=len(vectors))
            return True
            
        except Exception as e:
            logfire.error("Error upserting to Pinecone: {e}", e=e)
            return False
    
    def process_chunks_to_vectors(self, chunks, vectors_output_path):
        """Complete pipeline: generate embeddings, save vectors, and upsert to Pinecone"""
        vectors = self.generate_embeddings_for_chunks(chunks)
        
        if vectors:
            self.save_vectors(vectors, vectors_output_path)
            dimension = len(vectors[0]["values"])
            success = self.upsert_vectors_to_pinecone(vectors, dimension)
            
            if not success:
                return False
        else:
            logfire.error("No vectors generated")
            return False
        
        logfire.info("All {len} chunks processed successfully!", len=len(chunks))
        
        return True
    
    def find_similar_chunks(self, query_embedding, top_k=5):
        """Find similar chunks using Pinecone vector search"""
        try:
            index = self.pc.Index(self.pinecone_index_name)
            results = index.query(
                namespace=PINECONE_NAMESPACE,
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            similar_chunks = []
            for match in results.matches:
                similar_chunks.append({
                    'id': match.id,
                    'score': match.score,
                })
            
            return similar_chunks
            
        except Exception as e:
            logfire.error("Error finding similar chunks: {e}", e=e)
            return []
    
