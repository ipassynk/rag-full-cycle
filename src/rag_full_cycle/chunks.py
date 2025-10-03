import json
import logfire


class Chunks:
    def __init__(self, size, overlap):
        self.size = size
        self.overlap = overlap

    def create_chunks(self, extract_data):
        """Create chunks from extracted text data, respecting word boundaries"""
        chunks = []

        for data in extract_data:
            text = data["text"]
            words = text.split()
            step_size = self.size - self.overlap
            for i in range(0, len(words), step_size):
                chunk_words = words[i:i + self.size]
                chunk_text = " ".join(chunk_words)

                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "id": f"{data['page']}-{i // step_size}"
                    })

        return chunks

    def save_chunks(self, chunks, output_path):
        """Save chunks to file"""
        print(f"Saving {len(chunks)} chunks to {output_path}")
        with open(output_path, "w") as f:
            json.dump(chunks, f, indent=2)
        print(f"Saved chunks to {output_path}")

    def create_and_save_chunks(self, extract_data, output_path):
        """Create chunks and save to file"""
        chunks = self.create_chunks(extract_data)
        self.save_chunks(chunks, output_path)
        return chunks
