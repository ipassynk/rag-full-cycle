import os
import sys
import json
from pathlib import Path

def load_json_config(dataset="fy10"):
    """Load configuration from JSON files"""
    config_dir = Path(__file__).parent.parent.parent / "configs"
    
    base_config_path = config_dir / "base.json"
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    dataset_config_path = config_dir / f"{dataset}.json"
    with open(dataset_config_path, 'r') as f:
        dataset_config = json.load(f)
    
    config = {**base_config, **dataset_config}
    
    def expand_env_vars(obj):
        if isinstance(obj, dict):
            return {k: expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, str):
            return os.path.expandvars(obj)
        else:
            return obj
    
    config = expand_env_vars(config)
    
    return config

def get_dataset_from_args():
    """Extract dataset from command line arguments"""
    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
        if dataset in ["fy10", "baby"]:
            return dataset
    return "fy10"  # default

# Load the appropriate configuration and export variables
DATASET = get_dataset_from_args()
config = load_json_config(DATASET)

# Extract specific config values for backward compatibility
# Resolve PDF path relative to project root
pdf_file = config['dataset']['pdf_file']
if not os.path.isabs(pdf_file):
    # Get project root directory (parent of src directory)
    project_root = Path(__file__).parent.parent.parent
    PDF_FILE_PATH = str(project_root / pdf_file)
else:
    PDF_FILE_PATH = pdf_file
EXTRACT_OUTPUT = config['dataset']['extract_output']
CHUNK_SIZES = config['chunking']['sizes']
CHUNK_OVERLAPS = config['chunking']['overlaps']
QUESTION_GENERATION_PROMPT = config['question_generation']['prompt']
FILE_NAME = pdf_file.split('/')[-1].replace('.pdf', '').replace('_', '-').lower()
OUTPUT_DIR = f"{config['output_dir']}/{FILE_NAME}"
OPENAI_API_KEY = config['api_keys']['openai']
PINECONE_API_KEY = config['api_keys']['pinecone']
OPENROUTER_API_KEY = config['api_keys']['openrouter']
LOGFIRE_API_KEY = config['api_keys']['logfire']
TOP_K_RESULTS = config['vector_search']['top_k_results']
PINECONE_NAMESPACE = config['vector_search']['pinecone_namespace']
BATCH_SIZE = config['rate_limiting']['batch_size']
DELAY_BETWEEN_REQUESTS = config['rate_limiting']['delay_between_requests']
DELAY_BETWEEN_BATCHES = config['rate_limiting']['delay_between_batches']
MAX_RETRIES = config['rate_limiting']['max_retries']
QUESTION_MODEL = config['models']['question_model']
EMBEDDING_MODEL = config['models']['embedding_model']