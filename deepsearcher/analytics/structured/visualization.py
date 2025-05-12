# deepsearcher/analytics/visualization.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, List
from deepsearcher.analytics.base import AnalyticsBase

def plot_structured_data(
    analytics_base: AnalyticsBase,
    query: str,
    plot_type: str = "histogram",
    columns: Optional[List[str]] = None,
    title: Optional[str] = None,
    color: str = "blue"
) -> None:
    """
    Visualize structured data from a SQL query in a Streamlit GUI, with graceful error handling.

    Args:
        analytics_base (AnalyticsBase): Instance of AnalyticsBase for accessing data.
        query (str): SQL query to fetch structured data.
        plot_type (str): Type of plot ('histogram', 'boxplot', 'scatter'). Defaults to 'histogram'.
        columns (Optional[List[str]]): List of columns to visualize. If None, uses all columns.
        title (Optional[str]): Custom title for the plot. If None, uses a default title.
        color (str): Color for the plot elements. Defaults to 'blue'.
    """
    # Set Seaborn style for clean visuals
    sns.set(style="whitegrid")

    # Fetch data with error handling
    try:
        df = analytics_base.query_database(query)
        if df.empty:
            st.warning("The query returned no data. Please check your query.")
            return
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return

    # Filter columns if specified
    if columns:
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            st.error(f"Columns not found in data: {missing_columns}")
            return
        df = df[columns]

    # Generate the plot with error handling
    try:
        fig, ax = plt.subplots(figsize=(8, 6))

        if plot_type == "histogram":
            if len(df.columns) == 1:
                sns.histplot(df[df.columns[0]], color=color, ax=ax)
            else:
                st.error("Histogram requires exactly one column.")
                return
        elif plot_type == "boxplot":
            sns.boxplot(data=df, color=color, ax=ax)
        elif plot_type == "scatter":
            if len(df.columns) >= 2:
                sns.scatterplot(x=df.columns[0], y=df.columns[1], data=df, color=color, ax=ax)
            else:
                st.error("Scatter plot requires at least two columns.")
                return
        else:
            st.error(f"Unsupported plot type: '{plot_type}'. Use 'histogram', 'boxplot', or 'scatter'.")
            return

        # Set title
        default_title = f"{plot_type.capitalize()} for Query: {query}"
        ax.set_title(title if title else default_title, fontsize=14, pad=10)

        # Show plot in Streamlit
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error generating plot: {str(e)}")