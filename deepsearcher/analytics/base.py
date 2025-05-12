# deepsearcher/analytics/base.py
"""
# For structured data: config-->db instance-->results
db_config = {"connection_string": "sqlite:///my_database.db"}
analytics_structured = AnalyticsBase(data_source="structured", db_config=db_config)
results = analytics_structured.query_database("SELECT AVG(sales) FROM orders") 
"""

from typing import Optional, Literal

class AnalyticsBase:
    def __init__(
        self,
        data_source: Literal["structured"],
        vector_db=None,
        db_config: Optional[dict] = None
    ):
        """
        Initialize analytics for vector or structured data.

        Args:
            data_source (str): "vector" for Milvus, "structured" for SQL/Snowflake.
            vector_db: Vector database instance (required for vector data).
            db_config (dict, optional): Config for structured database connection.
        """
        self.data_source = data_source
        if data_source == "structured":
            if not db_config:
                raise ValueError("Database config is required for structured data.")
            self.db_config = db_config
            from .structured.database import DatabaseConnection
            self.db_connection = DatabaseConnection(db_config)
        else:
            raise ValueError("Invalid data_source. Use 'vector' or 'structured'.")

    # Vector data method
    def get_vectors(self, collection_name: str):
        if self.data_source != "vector":
            raise ValueError("This method is for vector data only.")
        return self.vector_db.query(collection_name)

    # Structured data method
    def query_database(self, query: str):
        if self.data_source != "structured":
            raise ValueError("This method is for structured data only.")
        return self.db_connection.execute_query(query)