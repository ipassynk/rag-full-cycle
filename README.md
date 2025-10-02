# RAG Full-Cycle Pipeline

A production-ready Retrieval-Augmented Generation (RAG) system built with modern AI technologies.

## Technologies & Stack

- **Vector Database**: Pinecone for scalable similarity search
- **Embeddings**: OpenAI's text-embedding-3-small model
- **LLM**: OpenAI GPT-4 via OpenRouter for question generation
- **PDF Processing**: pdfplumber for document extraction
- **Structured Output**: Instructor for type-safe LLM responses
- **Retry Logic**: Tenacity for robust API handling
- **Python**: Modern async/await patterns with Poetry dependency management

## Architecture

**End-to-End RAG Pipeline:**
1. **Document Processing**: PDF text extraction and intelligent chunking
2. **Vector Generation**: OpenAI embeddings with batch processing
3. **Vector Storage**: Pinecone serverless for production scalability
4. **Question Generation**: AI-powered question creation from content
5. **Retrieval Testing**: Automated similarity search validation

## Key Features

- **Multi-chunk Strategy**: Configurable chunk sizes and overlap
- **Batch Processing**: Optimized for large document sets
- **Error Handling**: Comprehensive retry mechanisms
- **Type Safety**: Pydantic models for data validation
- **Production Ready**: Rate limiting, error recovery, monitoring

## Quick Start

```bash
poetry install
poetry run pipeline
```
