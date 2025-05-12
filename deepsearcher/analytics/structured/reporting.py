# deepsearcher/analytics/reporting.py
from deepsearcher.analytics.base import AnalyticsBase
from deepsearcher import configuration
from deepsearcher.analytics.structured.statistics import calculate_structured_statistics
from typing import Optional, List

def generate_structured_report(
    analytics_base: AnalyticsBase,
    query: str,
    columns: Optional[List[str]] = None
) -> str:
    """
    Generate a textual report summarizing the statistics of structured data.

    Args:
        analytics_base (AnalyticsBase): Instance of AnalyticsBase.
        query (str): SQL query to fetch structured data.
        columns (Optional[List[str]]): List of column names to include in the report.

    Returns:
        str: A textual report summarizing the data statistics or a warning/error message.
    """
    # Attempt to calculate statistics from the structured data
    try:
        stats = calculate_structured_statistics(analytics_base, query, columns)
    except Exception as e:
        return f"Error: Failed to compute statistics - {str(e)}"

    # Check for warnings or errors in the statistics result
    if "warning" in stats:
        return f"Warning: {stats['warning']}"
    elif "error" in stats:
        return f"Error: {stats['error']}"

    # Generate a basic text report from the statistics
    report = f"Structured Data Report for Query: {query}\n\n"
    for col, metrics in stats.items():
        report += f"Column: {col}\n"
        for metric, value in metrics.items():
            report += f"  {metric}: {value}\n"
        report += "\n"

    # Enhance with LLM if available
    if configuration.llm:
        try:
            prompt = f"Provide a natural language summary of the following data statistics:\n{report}"
            enhanced_report = configuration.llm.chat(prompt)
            report = enhanced_report.content if enhanced_report else report
        except Exception as e:
            report += f"\nNote: LLM enhancement failed - {str(e)}"

    return report