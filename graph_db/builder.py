import os
import sys
from neo4j import GraphDatabase
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "food_business_db")

# Initialize MongoDB connection
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

class Neo4jConnection:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=30,
            connection_timeout=10  # Increased timeout for AuraDB
        )
    
    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return list(result)  # Consume results immediately

def build_graph():
    print("Initializing Neo4j connection to AuraDB...")
    neo4j = Neo4jConnection()
    
    # Test connection with proper result consumption
    try:
        test_query = "RETURN 1 AS test_value"
        result = neo4j.execute_query(test_query)
        if result and result[0]['test_value'] == 1:
            print("✓ Neo4j connection successful")
        else:
            raise ConnectionError("Neo4j connection test failed")
    except Exception as e:
        print(f"Failed to connect to Neo4j AuraDB: {str(e)}")
        print("Please verify:")
        print(f"1. Your AuraDB instance is running at: {NEO4J_URI}")
        print("2. IP whitelisting is properly configured")
        print("3. Credentials are correct in .env file")
        return None

    # Clear existing data
    print("Clearing existing graph data...")
    neo4j.execute_query("MATCH (n) DETACH DELETE n")
    
    # Create products
    print("Loading products...")
    products = {}
    inventory_count = db['inventories'].count_documents({})
    print(f"Found {inventory_count} products to load")
    
    for i, prod in enumerate(db['inventories'].find(), 1):
        query = """
        CREATE (p:Product {
            name: $name,
            price: $price,
            discount_price: $discount_price,
            quantity: $quantity,
            category: $category
        })
        RETURN id(p) as id
        """
        try:
            result = neo4j.execute_query(query, {
                "name": prod.get('productName', 'Unknown'),
                "price": float(prod.get('price', 0)),
                "discount_price": float(prod.get('productPrice', 0)),
                "quantity": int(prod.get('quantity', 0)),
                "category": prod.get('inventoryCategoryId', 'Uncategorized')
            })
            if result:
                products[prod['productName']] = result[0]['id']
                if i % 100 == 0:
                    print(f"Processed {i}/{inventory_count} products")
        except Exception as e:
            print(f"Error creating product {prod.get('productName')}: {str(e)}")
    
    # Create shops
    print("\nLoading shops...")
    shops = {}
    shop_count = db['shops'].count_documents({})
    print(f"Found {shop_count} shops to load")
    
    for shop in db['shops'].find():
        query = """
        CREATE (s:Shop {
            name: $name,
            address: $address,
            phone: $phone,
            delivery_charge: $delivery_charge,
            service_charge: $service_charge
        })
        RETURN id(s) as id
        """
        try:
            result = neo4j.execute_query(query, {
                "name": shop.get('shopName', 'Unknown Shop'),
                "address": shop.get('shopAddress', ''),
                "phone": shop.get('phoneNumber', ''),
                "delivery_charge": float(shop.get('deliveryCharge', 0)),
                "service_charge": float(shop.get('serviceCharge', 0))
            })
            if result:
                shops[shop['shopName']] = result[0]['id']
        except Exception as e:
            print(f"Error creating shop {shop.get('shopName')}: {str(e)}")
    
    # Create relationships
    print("\nCreating seller relationships...")
    invoice_count = db['invoiceitems'].count_documents({})
    print(f"Processing {invoice_count} invoices for relationships")
    
    for invoice in db['invoiceitems'].find():
        product_name = invoice.get('productName')
        shop_name = invoice.get('shopName')
        
        if product_name in products and shop_name in shops:
            query = """
            MATCH (s:Shop), (p:Product)
            WHERE id(s) = $shop_id AND id(p) = $product_id
            MERGE (s)-[r:SELLS]->(p)
            SET r.since = datetime()
            """
            try:
                neo4j.execute_query(query, {
                    "shop_id": shops[shop_name],
                    "product_id": products[product_name]
                })
            except Exception as e:
                print(f"Error linking {shop_name} -> {product_name}: {str(e)}")
    
    print("\nGraph build completed successfully!")
    print(f"Created: {len(products)} products, {len(shops)} shops")
    return neo4j

if __name__ == "__main__":
    build_graph()