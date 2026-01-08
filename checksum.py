import requests
import sys
import hashlib

HIDDEN_KEY = "67116e031024e38e146c9c61284d748f220376e109d941865c19d7d43f07a0e3"

def validate_checksum():
    # URL to the version file on GitHub
    url = "https://raw.githubusercontent.com/succulent94orange/Titan_Analyst/main/checksum.py"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        # We look for the "HIDDEN_KEY" line on GitHub
        remote_key = None
        for line in response.text.splitlines():
            if "HIDDEN_KEY =" in line:
                # Extract the hash string inside the quotes
                remote_key = line.split('"')[1]
                break

        if remote_key != HIDDEN_KEY:
            print("CRITICAL: Authentication Error. Access Denied.")
            sys.exit(1)
            
        print("Checksum Verified. Access Granted.")

    except Exception:
        # We keep the error message vague so they don't know why it failed
        print("System Error: Check connection.")
        sys.exit(1)
