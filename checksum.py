import requests
import sys
import os

def check_for_updates():
    # 1. The RAW URL of your file on GitHub
    GITHUB_RAW_URL = "https://raw.githubusercontent.com/succulent94orange/Titan_Analyst/main/checksum.py"
    
    # 2. Get the path of the currently running script
    local_file_path = os.path.realpath(__file__)
    
    try:
        # Fetch the content from GitHub
        response = requests.get(GITHUB_RAW_URL, timeout=10)
        response.raise_for_status()
        remote_content = response.text.strip()

        # Read the local file content
        with open(local_file_path, 'r', encoding='utf-8') as f:
            local_content = f.read().strip()

        # 3. Compare the two
        if remote_content != local_content:
            print("--------------------------------------------------")
            print("CRITICAL ERROR: Version Mismatch!")
            print("The local script does not match the GitHub version.")
            print("The program will now refuse to work.")
            print("--------------------------------------------------")
            sys.exit(1) # Kill the program
        else:
            print("Verification Successful: Versions match.")

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to GitHub: {e}")
        sys.exit(1)

# Run the check at the very beginning
if __name__ == "__main__":
    check_for_updates()
    
    # YOUR ACTUAL PROGRAM LOGIC STARTS HERE
    print("Welcome to Titan Analyst. Program is running...")
