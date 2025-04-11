import os
from huggingface_hub import hf_hub_download

os.environ["HF_TOKEN"] = "token_here"  # Replace with your token

output_path = hf_hub_download(
    repo_id="TheBloke/Llama-2-13B-chat-GGUF", 
    filename="llama-2-13b-chat.Q4_K_M.gguf",
    local_dir="models",
    token=os.environ["HF_TOKEN"]
)
