from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("PERSONAL_OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODELS_8 = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "qwen/qwen3-32b",
    "qwen/qwen3-14b",
    "google/gemma-3-27b-it",
    "google/gemma-3-12b-it",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
]