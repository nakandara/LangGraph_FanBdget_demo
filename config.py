#config.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# MongoDB Config
MONGO_URI = 'mongodb://localhost:27017'
DB_NAME = os.getenv("DB_NAME", "budget_app_db")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Neo4j Config
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://95c3c773.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "06OB6VgQQ7Fu8EU92d-wc0DYORDUctT9ZreYdNstfeY")