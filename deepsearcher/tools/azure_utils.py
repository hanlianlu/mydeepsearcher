# deepsearcher/utils/azure_utils.py
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import pandas as pd
from io import BytesIO
import os
import logging

log = logging.getLogger(__name__)

def fetch_full_tabular_data(blob_path: str, sheet_name: str = None) -> pd.DataFrame:
    """
    Fetch an Excel file from Azure Blob Storage using the account URL and secure authentication.

    :param blob_path: Full blob path (e.g., 'sandbox/data.xlsx').
    :return: pandas DataFrame with the file's data.
    """
    try:
        # Get account URL from .env
        account_url = os.getenv("AZURE_BLOB_ACCOUNT_URL")
        if not account_url:
            raise ValueError("AZURE_BLOB_ACCOUNT_URL not set in environment.")

        # Use DefaultAzureCredential for secure authentication
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

        # Split blob_path into container and blob name
        container_name, blob_name = blob_path.split("/", 1)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        # Download blob to memory and load into DataFrame
        blob_data = blob_client.download_blob().readall()
        df = pd.read_excel(BytesIO(blob_data), sheet_name=sheet_name)
        log.info(f"Successfully fetched tabular data from {blob_path}")
        return df
    except Exception as e:
        log.error(f"Failed to fetch data from {blob_path}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure