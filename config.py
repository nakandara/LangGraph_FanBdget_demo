#config.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = 'mongodb://localhost:27017'
DB_NAME = os.getenv("DB_NAME","budget_app_db")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
