#vector_store.py
import os
from config import db, GEMINI_API_KEY
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # ✅ Updated line
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings


DB_PATH = "faiss_index"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30

def build_vector_db():
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Process inventories collection for product data
    inventory_collection = db['inventories']
    for product in inventory_collection.find({}, {"_id": 0}):
        text_parts = [
            f"Product: {product.get('productName', '')}",
            f"Type: {product.get('productType', '')}",
            f"Brand: {product.get('brandName', '')}",
            f"Regular Price: {product.get('price', 'N/A')}",
            f"Discount Price: {product.get('productPrice', 'N/A')}",
            f"Discount: {product.get('productDiscount', '0')}{'%' if product.get('discountType') == 'PERCENTAGE' else ' LKR'}",
            f"Category: {product.get('inventoryCategoryId', '')}",
            f"Quantity: {product.get('quantity', '')} {product.get('productType', '')}"
        ]
        
        text = "\n".join(text_parts)
        metadata = {
            "collection": "inventories",
            "product_name": product.get('productName', ''),
            "price": product.get('price', ''),
            "source": "mongodb"
        }
        documents.append(Document(page_content=text, metadata=metadata))
        print(f"Added product: {product.get('productName', '')}")  # Debug

    # Process invoiceitems collection for historical sales data
    invoice_items_collection = db['invoiceitems']
    for item in invoice_items_collection.find({}, {"_id": 0}):
        text_parts = [
            f"Item: {item.get('productName', '')}",
            f"Sold Price: {item.get('price', 'N/A')}",
            f"Quantity Sold: {item.get('quantity', '')}",
            f"Total Amount: {item.get('amount', 'N/A')}",
            f"Original Price: {item.get('productPrice', 'N/A')}",
            f"Discount: {item.get('productDiscount', '0')}{'%' if item.get('discountType') == 'PERCENTAGE' else ' LKR'}"
        ]
        
        text = "\n".join(text_parts)
        metadata = {
            "collection": "invoiceitems",
            "product_name": item.get('productName', ''),
            "price": item.get('price', ''),
            "source": "mongodb"
        }
        documents.append(Document(page_content=text, metadata=metadata))
    # Process users collection
    users_collection = db['users']
    for user in users_collection.find({}, {"_id": 0, "password": 0, "firebaseToken": 0, "__v": 0}):  # Exclude sensitive fields
        text_parts = [
            f"User Name: {user.get('name', 'N/A')}",
            f"Email: {user.get('email', 'N/A')}",
            f"Phone: {user.get('phoneNumber', 'N/A')}",
            f"User Type: {user.get('userType', 'N/A')}",
            f"Role: {user.get('role', 'N/A')}",
            f"Premium Status: {user.get('premiumStatus', 'N/A')}",
            f"Premium Type: {user.get('premiumUserType', 'N/A')}",
            f"Verification Status: {user.get('verifiedStatus', 'N/A')}",
            f"Account Medium: {user.get('medium', 'N/A')}",
            f"Manages Inventory: {'Yes' if user.get('isMaintainInventory', False) else 'No'}"
        ]
        
        text = "\n".join([part for part in text_parts if not part.endswith('N/A')])
        metadata = {
            "collection": "users",
            "user_name": user.get('name', ''),
            "email": user.get('email', ''),
            "type": "user_profile",
            "source": "mongodb"
        }
        documents.append(Document(page_content=text, metadata=metadata))
        print(f"Added user: {user.get('name', '')}")  # Debug


     # Process shops collection
    shops_collection = db['shops']
    for shop in shops_collection.find({}, {"_id": 0, "__v": 0}):  # Exclude _id and __v
        text_parts = [
            f"Shop Name: {shop.get('shopName', 'N/A')}",
            f"Owner: {shop.get('ownerName', 'N/A')}",
            f"Address: {shop.get('shopAddress', 'N/A')}",
            f"Phone: {shop.get('phoneNumber', 'N/A')}",
            f"Service Charge: {shop.get('serviceCharge', '0')} {shop.get('serviceChargeType', 'LKR')}",
            f"Delivery Charge: {shop.get('deliveryCharge', '0')} LKR",
            f"Note: {shop.get('shortNote', '')}"
        ]
        
        text = "\n".join([part for part in text_parts if not part.endswith('N/A')])
        metadata = {
            "collection": "shops",
            "shop_name": shop.get('shopName', ''),
            "type": "shop_info",
            "source": "mongodb"
        }
        documents.append(Document(page_content=text, metadata=metadata))
        print(f"Added shop: {shop.get('shopName', '')}")  # Debug


    print(f"Total documents before splitting: {len(documents)}")  # Debug
    chunks = splitter.split_documents(documents)
    print(f"Total chunks after splitting: {len(chunks)}")  # Debug
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=GEMINI_API_KEY  # Use from config
    )
    
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(DB_PATH)
    return vector_db
def load_vector_db():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        return build_vector_db()
