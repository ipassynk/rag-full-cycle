import json
from .config import *


class ChunkGenerator:
    def __init__(self):
        pass
    
    def create_chunks(self, extract_data, size, overlap):
        """Create chunks from extracted text data"""
        chunks = []
        
        for data in extract_data:
            text = data["text"]
            window_chunks = [text[i:i+size] for i in range(0, len(text), size-overlap)]
            for chunk_num, chunk in enumerate(window_chunks, 0):
                chunks.append({
                    "text": chunk, 
                    "id": f"{data['page']}-{chunk_num}"
                })
        
        return chunks
    
    def save_chunks(self, chunks, output_path):
        """Save chunks to file"""
        print(f"Saving {len(chunks)} chunks to {output_path}")
        with open(output_path, "w") as f:
            json.dump(chunks, f, indent=2)
        print(f"Saved chunks to {output_path}")
    
    def create_and_save_chunks(self, extract_data, size, overlap, output_path):
        """Create chunks and save to file"""
        chunks = self.create_chunks(extract_data, size, overlap)
        self.save_chunks(chunks, output_path)
        return chunks
    
    def process_all_chunk_sizes(self, extract_data):
        """Process all chunk sizes and overlaps from config"""
        print("\nCreating chunks...")
        
        all_chunks = {}
        
        for num, size in enumerate(CHUNK_SIZES, 0):
            overlap = CHUNK_OVERLAPS[num]
            print(f"Processing chunk size: {size}, overlap: {overlap}")
            
            chunks_output = f"{OUTPUT_DIR}/chunks-{size}-{overlap}.json"
            chunks = self.create_and_save_chunks(extract_data, size, overlap, chunks_output)
            all_chunks[f"{size}-{overlap}"] = chunks
        
        return all_chunks
