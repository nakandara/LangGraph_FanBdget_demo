from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

from agent_graph import build_graph, explain_chain  # make sure explain_chain is exported from agent_graph

app = FastAPI()

# Allow requests from your frontend (adjust if needed)
origins = [
    "http://localhost:3000",  # React/Vite/Next frontend
    "http://127.0.0.1:3000",
    "*"  # Optional: allow all origins (not recommended for production)
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,             # Origins allowed to access the backend
    allow_credentials=True,
    allow_methods=["*"],               # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],               # Allow all headers
)

# Build the LangGraph
graph = build_graph()

# Pydantic model for request body
class QuestionInput(BaseModel):
    question: str

# Route to handle LangGraph question
@app.post("/ask")
async def ask_question(input: QuestionInput):
    try:
        print(f"📥 Received question: {input.question}")
        result = await asyncio.to_thread(graph.invoke, {"question": input.question})
        print("✅ Final answer generated.")
        return {
            "question": input.question,
            "answer": result.get("response", "No response generated")
        }
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return {"error": str(e)}

# Test route to verify explain_chain works alone
@app.get("/test")
def test_explain_chain():
    try:
        print("🧪 Testing explain_chain with sample input...")
        response = explain_chain.invoke({"data": "Test Sinhala data"})
        print("✅ Explanation generated.")
        return {"response": response}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}
