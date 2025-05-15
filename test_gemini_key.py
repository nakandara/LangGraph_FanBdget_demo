# test_gemini_key.py
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key="AIzaSyBS-HNAeY6M00PDRSPOUHLZEkquxzwL3iY"
)

response = llm.invoke("What is the capital of France?")
print("✅ Response from Gemini:", response)
