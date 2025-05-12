import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from deepsearcher.analytics.base import AnalyticsBase

def calculate_structured_statistics(
    analytics_base: AnalyticsBase,
    query: str,
    columns: Optional[List[str]] = None
) -> Dict[str, Union[Dict[str, float], str]]:
    """
    Calculate statistical metrics for numerical columns based on a SQL query.

    Args:
        analytics_base (AnalyticsBase): Instance of AnalyticsBase for accessing the data source.
        query (str): SQL query to fetch structured data (e.g., "SELECT * FROM table_name").
        columns (Optional[List[str]]): List of column names to analyze. If None, all columns are considered.

    Returns:
        Dict[str, Union[Dict[str, float], str]]: A dictionary where each key is a column name and the value is a dictionary
                                                 of statistical metrics for numerical columns. If no numerical columns are
                                                 found or an issue occurs, returns a dictionary with a warning or error message.
    """
    # Step 1: Fetch data from the database
    try:
        result = analytics_base.query_database(query)
    except Exception as e:
        return {"warning": f"Failed to execute query: {str(e)}"}

    # Step 2: Ensure the result is a pandas DataFrame
    if isinstance(result, pd.DataFrame):
        df = result
    else:
        try:
            # Attempt to convert the result (e.g., list of tuples) to a DataFrame
            df = pd.DataFrame(result)
        except Exception as e:
            return {"error": f"Failed to convert query result to DataFrame: {str(e)}"}

    # Step 3: Check if the DataFrame is empty
    if df.empty:
        return {"warning": "The query returned an empty result set."}

    # Step 4: Filter columns if specified
    if columns is not None:
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            return {"error": f"Specified columns not found in the data: {missing_columns}"}
        df = df[columns]

    # Step 5: Identify numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Step 6: Check for numerical columns and return a warning if none exist
    if not numerical_columns:
        return {
            "warning": "No numerical columns found in the query result or specified columns. "
                       "Statistics are only computed for numerical data."
        }

    # Step 7: Compute statistics for numerical columns
    statistics = {}
    for col in numerical_columns:
        col_data = df[col]
        statistics[col] = {
            "mean": col_data.mean(),
            "std": col_data.std(),
            "min": col_data.min(),
            "max": col_data.max(),
            "median": col_data.median(),
            "count": col_data.count(),
            "sum": col_data.sum(),
            "distinct_count": col_data.nunique()  # Updated key name for consistency
        }

    # Step 8: Return the computed statistics
    return statistics