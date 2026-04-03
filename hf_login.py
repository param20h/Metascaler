"""
Interactive HuggingFace login script
Usage: python hf_login.py
"""
from huggingface_hub import login
import os

print("=" * 60)
print("HuggingFace Hub Login")
print("=" * 60)
print("\nYou can authenticate in two ways:")
print("1. Enter your API token interactively")
print("2. Set HF_TOKEN environment variable and run with --auto flag")
print("\nTo get a token, visit: https://huggingface.co/settings/tokens")
print("=" * 60)

token = os.getenv("HF_TOKEN", "").strip()

if token:
    print(f"\nUsing token from HF_TOKEN environment variable...")
    try:
        login(token=token)
        print("✓ Login successful!")
    except Exception as e:
        print(f"✗ Login failed: {e}")
else:
    print("\nEnter your HuggingFace token (or type 'quit' to exit):")
    token = input("> ").strip()
    if token.lower() != 'quit':
        try:
            login(token=token)
            print("✓ Login successful!")
        except Exception as e:
            print(f"✗ Login failed: {e}")
    else:
        print("Login cancelled.")
