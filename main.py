from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from agent_graph import build_graph, explain_chain  # Now these imports will work

app = FastAPI()

# CORS setup (keep your existing CORS configuration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the graph
chain = build_graph()

class QuestionInput(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(input: QuestionInput):
    try:
        print(f"📥 Received question: {input.question}")
        result = await asyncio.to_thread(
            chain.invoke,
            {"question": input.question}
        )
        return {
            "question": input.question,
            "answer": result.get("final_answer", "No response generated")
        }
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return {"error": str(e)}

@app.get("/test")
async def test_chain():
    try:
        test_response = explain_chain.invoke({
            "question": "Test question",
            "semantic_results": "Test data",
            "keyword_results": "Test keywords",
            "graph_results": "Test graph",
            "current_date": "2025-05-17"
        })
        return {"response": test_response.content}
    except Exception as e:
        return {"error": str(e)}