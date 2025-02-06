from huggingface_hub import HfApi

def list_repo_files():
    repo_id = 'unsloth/DeepSeek-R1-GGUF'
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id)
        print(f"Files in {repo_id}:")
        for f in files:
            print(f)
    except Exception as e:
        print(f"Failed to list repository files: {e}")

if __name__ == "__main__":
    list_repo_files()