import requests
import sys

# The version this file expects
VERSION = 1997 

def validate_checksum():
    # URL to the version file on GitHub 
    # (You can point this to main.py or a dedicated version.txt file)
    url = "https://raw.githubusercontent.com/succulent94orange/Titan_Analyst/main/checksum.py"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        # Search for the VERSION line in the GitHub file
        remote_version = None
        for line in response.text.splitlines():
            if "VERSION =" in line:
                # Extract the number from the string
                remote_version = int(''.join(filter(str.isdigit, line)))
                break

        if remote_version != VERSION:
            print(f"CRITICAL: Version Mismatch (Local: {VERSION} | Remote: {remote_version})")
            print("This program is locked. Please contact the administrator.")
            sys.exit(1) # This stops the entire program
            
        print("Checksum Verified. Access Granted.")
        return True

    except Exception as e:
        print(f"Error: Could not reach verification server. {e}")
        sys.exit(1)
