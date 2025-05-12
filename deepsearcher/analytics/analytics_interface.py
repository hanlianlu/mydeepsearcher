


# deepsearcher/analytics_interface.py
from typing import Optional, Dict, Literal
from deepsearcher.analytics.base import AnalyticsBase

from deepsearcher.analytics.structured.statistics import calculate_structured_statistics
from deepsearcher.analytics.structured.visualization import plot_structured_data
from deepsearcher.analytics.structured.reporting import generate_structured_report

def perform_analytics(
    data_source: Literal["structured"],
    collection_name: Optional[str] = None,  # For vector data
    table_name: Optional[str] = None,       # For structured data
    compute_statistics: bool = True,
    visualize: bool = False,
    generate_report: bool = False,
    db_config: Optional[dict] = None        # For structured data, {"connection_string": "sqlite:///my_database.db"}
) -> Dict[str, any]:
    """
    Perform analytics on vectorized or structured data.

    Args:
        data_source (str): "vector" for Milvus, "structured" for SQL/Snowflake.
        collection_name (str, optional): Milvus collection name (for vector data).
        table_name (str, optional): Database table name (for structured data).
        compute_statistics (bool): Compute statistics for the data.
        visualize (bool): Visualize the data (embeddings for vector, plots for structured).
        generate_report (bool): Generate a report based on the analysis.
        db_config (dict, optional): Database configuration for structured data (e.g., connection string).

    Returns:
        Dict[str, any]: Results of the requested analytics tasks.

    Raises:
        ValueError: If required parameters are missing or data_source is invalid.
    """
    # Validate data_source
    if data_source not in ["vector", "structured"]:
        raise ValueError("Invalid data_source. Use 'vector' or 'structured'.")

    # Structured data processing
    elif data_source == "structured":
        if not table_name or not db_config:
            raise ValueError("table_name and db_config are required for structured data.")
        analytics_db_instance = AnalyticsBase(data_source="structured", db_config=db_config)
        results = {}
        if compute_statistics:
            results["statistics"] = calculate_structured_statistics(analytics_db_instance, table_name)
        if visualize:
            plot_structured_data(analytics_db_instance, table_name)
        if generate_report:
            results["report"] = generate_structured_report(analytics_db_instance, table_name)
        return results


