
from datetime import datetime
from config import db

class ProductNode:
    """Represents products in your inventory"""
    @staticmethod
    def create(graph, product_data):
        node = Node("Product",
                   name=product_data.get('productName', ''),
                   type=product_data.get('productType', ''),
                   brand=product_data.get('brandName', ''),
                   price=float(product_data.get('price', 0)),
                   discount_price=float(product_data.get('productPrice', 0)),
                   discount_type=product_data.get('discountType', 'NONE'),
                   quantity=int(product_data.get('quantity', 0)),
                   category=product_data.get('inventoryCategoryId', ''),
                   last_updated=datetime.utcnow())
        graph.create(node)
        return node

class ShopNode:
    """Represents physical shops/stores"""
    @staticmethod
    def create(graph, shop_data):
        node = Node("Shop",
                   name=shop_data.get('shopName', ''),
                   owner=shop_data.get('ownerName', ''),
                   address=shop_data.get('shopAddress', ''),
                   phone=shop_data.get('phoneNumber', ''),
                   service_charge=float(shop_data.get('serviceCharge', 0)),
                   delivery_charge=float(shop_data.get('deliveryCharge', 0)),
                   note=shop_data.get('shortNote', ''),
                   last_updated=datetime.utcnow())
        graph.create(node)
        return node

class UserNode:
    """Represents customers/users"""
    @staticmethod
    def create(graph, user_data):
        node = Node("User",
                   name=user_data.get('name', ''),
                   email=user_data.get('email', ''),
                   phone=user_data.get('phoneNumber', ''),
                   user_type=user_data.get('userType', 'REGULAR'),
                   premium_status=user_data.get('premiumStatus', False),
                   verified=user_data.get('verifiedStatus', False),
                   last_active=datetime.utcnow())
        graph.create(node)
        return node

class InvoiceNode:
    """Represents transaction records"""
    @staticmethod
    def create(graph, invoice_data):
        node = Node("Invoice",
                   invoice_id=invoice_data.get('invoiceId', ''),
                   total=float(invoice_data.get('totalAmount', 0)),
                   date=invoice_data.get('createdAt', datetime.utcnow()),
                   status=invoice_data.get('status', 'COMPLETED'))
        graph.create(node)
        return node

# Define relationship types
class Relationships:
    SELLS = lambda g, shop, product: Relationship(shop, "SELLS", product)
    PURCHASED = lambda g, user, product, invoice: Relationship(user, "PURCHASED", product,
        through=invoice,
        quantity=invoice.get('quantity', 1),
        price=float(invoice.get('price', 0)),
        date=invoice.get('createdAt', datetime.utcnow()))
    OWNS = lambda g, user, shop: Relationship(user, "OWNS", shop)
    RELATED = lambda g, prod1, prod2: Relationship(prod1, "RELATED_TO", prod2,
        similarity=0.8,  # Can be calculated from co-purchase data
        last_updated=datetime.utcnow())

# Export all relationships
RELATIONSHIPS = {
    'SELLS': Relationships.SELLS,
    'PURCHASED': Relationships.PURCHASED,
    'OWNS': Relationships.OWNS,
    'RELATED': Relationships.RELATED
}

# Helper function for batch creation
def create_nodes_batch(graph, collection_name, node_class):
    """Bulk create nodes from a MongoDB collection"""
    collection = db[collection_name]
    nodes = []
    for item in collection.find():
        nodes.append(node_class.create(graph, item))
    return nodes