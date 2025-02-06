from huggingface_hub import snapshot_download
import os

def download_model():
    repo_id = 'unsloth/DeepSeek-R1-GGUF'
    cache_dir = 'models'
    try:
        # Download the entire repository, preserving directory structure
        snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
        print(f"Successfully downloaded all files from '{repo_id}' to '{cache_dir}/'")
    except Exception as e:
        print(f"Failed to download model: {e}")

if __name__ == "__main__":
    download_model()