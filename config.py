from dotenv import load_dotenv
import os
load_dotenv()  # loads .env from current directory

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")