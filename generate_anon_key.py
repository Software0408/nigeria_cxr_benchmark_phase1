# generate_anon_key.py
# One-time script to generate anonymization secret key

import secrets

key = secrets.token_hex(32)  # 64-character hex
print("Your generated secret key: ", key)
print("Copy this, store securely (e.g., as env var)")




# echo $env:CXR_ANON_SECRET
# setx CXR_ANON_SECRET "your_generated_key_here"  # Windows PowerShell