# openai_setup.py

import os
from dotenv import load_dotenv
import openai

# Load variables from .env
load_dotenv()

# Set your API Key and Organization ID
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

# (Optional) Test it
try:
    models = openai.models.list()
    print("✅ OpenAI connection successful. Models available:")
    for model in models.data:
        print(" -", model.id)
except Exception as e:
    print("❌ Error connecting to OpenAI:", e)
