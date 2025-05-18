GRAPH_QUERIES = {
    "product_search": """
    MATCH (p:Product)
    WHERE toLower(p.name) CONTAINS toLower($query)
    RETURN p LIMIT 3
    """,
    "shop_products": """
    MATCH (s:Shop {name: $shop_name})-[:SELLS]->(p:Product)
    RETURN p
    """
}