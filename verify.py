import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face API key from the environment
inference_api_key = os.getenv("HUGGINGFACE_TOKEN")

# Initialize the embeddings with a different model
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key, model_name="BAAI/bge-small-en-v1.5"
)

# Test embedding a sentence
text = "This is a test sentence."
query_result = embeddings.embed_query(text)
print(query_result[:3])
