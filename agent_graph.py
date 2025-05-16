#agent_graph.py
from langgraph.graph import StateGraph
import asyncio
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from vector_store import load_vector_db
from langchain.retrievers import EnsembleRetriever
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain.retrievers import BM25Retriever
from config import db

load_dotenv()

import os

api_key = os.getenv("GEMINI_API_KEY")

# Define the state schema
class GraphState(TypedDict):
    question: str
    raw_data: List[str]
    final_answer: str
    response: str

# Initialize model and memory
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)
memory = ConversationBufferMemory(return_messages=True)

# Define prompt and chain using RunnableSequence
prompt = PromptTemplate.from_template("""
You are a knowledgeable assistant for a food business. Provide accurate, friendly responses based strictly on the retrieved data.

Guidelines:
1. For product queries:
- State current price and any discounts
- Mention measurement units if relevant
- Note availability when known
- Example: "Cheese Koththu: 1,750 LKR (regularly 1,100 LKR), sold as single portions"

2. For order/invoice questions:
- Summarize key details (items, totals, status)
- Example: "Your order #INV0001 includes 3 Cheese Koththu (5,250 LKR) with 100 toffees (1,800 LKR), total 7,050 LKR"

3. For shop information:
- Share relevant policies and contact details
- Example: "We charge 550 LKR for delivery. Call us at 077-6694351 for orders."

4. When uncertain:
- "I couldn't verify that information. Please call 077-6694351 for assistance."

Always:
- Use LKR for currency
- Keep responses clear and professional
- Maintain a helpful, welcoming tone
- Include specific numbers when available
- Group related items logically

Retrieved Data:
{data}

User Question: 
{question}

Provide the most complete, accurate response possible:
""")

explain_chain: RunnableSequence = prompt | llm

# --- Semantic retriever ---
vector_db = load_vector_db()
semantic_retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})


# --- Keyword retriever using BM25 ---
# Create basic BM25 retriever from raw mongo docs
def create_bm25_retriever():
    from langchain.schema import Document
    all_docs = []
    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        for doc in collection.find({}, {"_id": 0}):
            text = f"{doc.get('title', '')}\n{doc.get('description', '')}\n{doc.get('category', '')}"
            all_docs.append(Document(page_content=text, metadata={"collection": collection_name}))
    
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = 4
    return bm25

keyword_retriever = create_bm25_retriever()

# --- Combine both (Hybrid) ---
retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, keyword_retriever],
    weights=[0.6, 0.4]  # adjust weight as needed
)
print("🔗 Hybrid retriever initialized.",retriever)

# Step 1: Retrieve relevant documents
def semantic_retrieve_step(state: GraphState) -> GraphState:
    print("🔍 Retrieving documents...")
    question = state["question"]
    memory.chat_memory.add_user_message(question)
    docs = retriever.get_relevant_documents(question)
    formatted = [f"[collection: {doc.metadata.get('collection')}]\n{doc.page_content}" for doc in docs]
    print(f"✅ Retrieved {len(docs)} documents.")

    return {
        "question": question,
        "raw_data": formatted
    }

# Step 2: Use LLM to explain data
def explain_step(state: GraphState) -> GraphState:
    print("🧠 Generating explanation...")
    data = "\n\n".join(state["raw_data"])
    question = state["question"]

    response = asyncio.run(
        asyncio.to_thread(
            explain_chain.invoke,
            {"data": data, "question": question}
        )
    )

    memory.chat_memory.add_ai_message(response)
    print("✅ Got response from LLM.")

    return {
        "question": question,
        "raw_data": state["raw_data"],
        "final_answer": response
    }


# Step 3: Final formatting
def final_step(state: GraphState) -> GraphState:
    return {
        "question": state["question"],
        "raw_data": state["raw_data"],
        "final_answer": state["final_answer"],
        "response": state["final_answer"]
    }

# Build the graph
def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("retrieve", semantic_retrieve_step)
    builder.add_node("explain", explain_step)
    builder.add_node("final", final_step)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "explain")
    builder.add_edge("explain", "final")

    return builder.compile()

# agent_graph.py
__all__ = ["build_graph", "explain_chain"]