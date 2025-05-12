# deepsearcher/analytics/structured/database.py
from sqlalchemy import create_engine

class DatabaseConnection:
    def __init__(self, db_config: dict):
        """
        Initialize a connection to a structured database.

        Args:
            db_config (dict): Includes 'connection_string' (e.g., 'sqlite:///data.db').
        """
        self.engine = create_engine(db_config["connection_string"])
        self.connection = self.engine.connect()

    def execute_query(self, query: str):
        """
        Execute a SQL query.

        Args:
            query (str): SQL query to run.

        Returns:
            list: Query results.
        """
        try:
            result = self.connection.execute(query)
            return result.fetchall()
        except Exception as e:
            raise ValueError(f"Query failed: {e}")