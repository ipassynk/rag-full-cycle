import json
import re
import pdfplumber
import logfire

class Extracts:
    def __init__(self):
        pass
    
    def clean_text(self, text):
        """Clean extracted text by removing excessive whitespace and empty lines"""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)  
        text = text.strip()

        lines = text.split('\n')
        long_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 2:
                long_lines.append(line)
        
        cleaned_text = '\n'.join(long_lines)
        cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
        return cleaned_text.strip()
    
    def extract_text_from_pdf(self, pdf_path, max_pages=None):
        """Extract text from PDF file using pdfplumber"""
        logfire.info("Extracting text from PDF with pdfplumber...")
        
        extract = []
        with pdfplumber.open(pdf_path) as pdf:
            for num, page in enumerate(pdf.pages, 1):
                if max_pages and num >= max_pages:
                    break
                    
                raw_text = page.extract_text()
                cleaned_text = raw_text #self.clean_text(raw_text)
                
                if cleaned_text:
                    extract.append({"page": num, "text": cleaned_text})
                    logfire.info("Page {num}: {len} characters", num=num, len=len(cleaned_text))
        
        logfire.info("Extracted text from {len} pages", len=len(extract))
        return extract
    
    def save_extract(self, extract_data, output_path):
        """Save extracted text to file"""
        logfire.info("Saving extracted text...")
        with open(output_path, "w") as f:
            json.dump(extract_data, f, indent=2)
        logfire.info("Saved extracted text to {output_path}", output_path=output_path)
    
    def extract_and_save(self, pdf_path, output_path, max_pages=None):
        """Extract text from PDF and save to file"""
        extract_data = self.extract_text_from_pdf(pdf_path, max_pages)
        self.save_extract(extract_data, output_path)
        return extract_data
