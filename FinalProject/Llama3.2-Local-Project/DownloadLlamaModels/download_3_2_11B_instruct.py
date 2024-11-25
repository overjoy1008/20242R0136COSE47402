import os
from huggingface_hub import snapshot_download
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create Assets/Models directory if it doesn't exist
model_dir = Path("Assets/Models/llama_3_2_11B_instruct")
model_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
    use_auth_token=os.getenv("HUGGINGFACE_LLAMA_TOKEN"),
    local_dir="Assets/Models/llama_3_2_11B_instruct",
)