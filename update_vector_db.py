#update_vector_db.py
import os
from datetime import datetime, timedelta
from config import db, GEMINI_API_KEY
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


DB_PATH = "faiss_index"

def update_vector_db():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    if not os.path.exists(DB_PATH):
        print("Run full vector build first.")
        return

    vector_db = FAISS.load_local(DB_PATH, embeddings)
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    cutoff_time = datetime.utcnow() - timedelta(hours=1)

    new_documents = []
    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        recent_docs = collection.find({"last_updated": {"$gte": cutoff_time}}, {"_id": 0})
        for doc in recent_docs:
            text = f"{doc.get('title', '')}\n{doc.get('description', '')}\n{doc.get('category', '')}"
            metadata = {"collection": collection_name}
            new_documents.append(Document(page_content=text, metadata=metadata))

    if new_documents:
        chunks = splitter.split_documents(new_documents)
        vector_db.add_documents(chunks)
        vector_db.save_local(DB_PATH)
        print(f"✅ Updated with {len(chunks)} chunks.")
    else:
        print("🟡 No updates found.")
