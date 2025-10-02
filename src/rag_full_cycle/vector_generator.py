import json
import time
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from .config import *
from .embedding_generator import EmbeddingGenerator

class VectorGenerator:
    def __init__(self, size, overlap, embedding_model):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.embedding_model = embedding_model
        self.size = size
        self.overlap = overlap
        self.pinecone_index_name = f"{size}-{overlap}-{embedding_model}"
    
    def generate_embeddings_for_chunks(self, chunks):
        """Generate embeddings for a list of chunks"""
        print(f"\nGenerating embeddings for {len(chunks)} chunks...")
        print(f"Using batch size: {BATCH_SIZE}, delay: {DELAY_BETWEEN_REQUESTS}s")
        
        embedding_generator = EmbeddingGenerator(size=self.size, overlap=self.overlap, embedding_model=EMBEDDING_MODEL_3_SMALL)
        vectors_to_upsert = []
        
        # Process chunks in batches to improve performance
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(chunks))
            batch = chunks[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//BATCH_SIZE + 1}: chunks {batch_start+1}-{batch_end}")
            
            for i, chunk in enumerate(batch):
                try:
                    chunk_index = batch_start + i
                    print(f"  Processing chunk {chunk_index+1}/{len(chunks)}: {chunk['id']}")
                    
                    # Create embedding
                    response = embedding_generator.create_embedding_with_retry(chunk["text"])
                    embedding = response.data[0].embedding
                    vectors_to_upsert.append({
                        "id": chunk["id"], 
                        "values": embedding
                    })
                    
                    time.sleep(DELAY_BETWEEN_REQUESTS)
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk['id']}: {e}")
                    continue
            
            if batch_end < len(chunks):
                print(f"  Waiting {DELAY_BETWEEN_BATCHES}s before next batch...")
                time.sleep(DELAY_BETWEEN_BATCHES)
        
        return vectors_to_upsert
    
    def save_vectors(self, vectors, output_path):
        """Save vectors to file"""
        print(f"Saving {len(vectors)} vectors to {output_path}")
        with open(output_path, "w") as f:
            json.dump(vectors, f, indent=2)
        print(f"Saved vectors to {output_path}")
    
    def create_and_manage_index(self, vectors, dimension):
        """Create Pinecone index if needed and return index object"""
        print(f"Index {self.pinecone_index_name} creating one...")
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
            print(f"Upserting {len(vectors)} vectors to Pinecone, dimension: {dimension}...")
            
            index = self.create_and_manage_index(vectors, dimension)
            index.upsert(
                namespace=PINECONE_NAMESPACE,
                vectors=vectors
            )
            print(f"Successfully upserted {len(vectors)} vectors to Pinecone")
            return True
            
        except Exception as e:
            print(f"Error upserting to Pinecone: {e}")
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
            print("No vectors generated")
            return False
        
        print(f"All {len(chunks)} chunks processed successfully!")
        
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
            print(f"Error finding similar chunks: {e}")
            return []
    
