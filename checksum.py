import requests
import hashlib
import sys

def get_file_hash(content):
    """Generates a SHA-256 hash for the given content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def verify_and_run():
    # 1. Configuration
    github_url = "https://raw.githubusercontent.com/username/repo/main/config.json"
    local_file_path = "config.json"

    try:
        # 2. Fetch remote content
        response = requests.get(github_url)
        response.raise_for_status()
        remote_content = response.text
        remote_hash = get_file_hash(remote_content)

        # 3. Read local content
        with open(local_file_path, "r") as f:
            local_content = f.read()
            local_hash = get_file_hash(local_content)

        # 4. Comparison
        if remote_hash != local_hash:
            print("ERROR: Local file does not match the GitHub version.")
            print("Please update your file before running the program.")
            sys.exit(1) # Stop the program
            
        print("Success: Files match. Launching program...")
        # Your main logic goes here
        
    except FileNotFoundError:
        print(f"ERROR: Local file {local_file_path} not found.")
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    verify_and_run()
