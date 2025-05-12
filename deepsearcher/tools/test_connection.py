import os
import yaml
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

def load_config():
    """
    Load the configuration from config.yaml located at the repository root.
    Assumes this script is in deepsearcher/examples.
    """
    config_path = os.path.join(os.path.dirname(__file__), "../..", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def initialize_client(config, service_type):
    """
    Initialize the AzureOpenAI client based on the service type (llm or embedding).
    Returns a tuple: (client, deployment_model, endpoint, api_version)
    """
    service_config = config.get("provide_settings", {}).get(service_type, {}).get("config", {})
    # Support either "azure_endpoint" or "base_url" for embedding
    endpoint = service_config.get("azure_endpoint") or service_config.get("base_url")
    api_key = os.environ.get( "AZURE_OPENAI_API_KEY" , service_config.get("api_key"))
    # For embedding, if api_version is not provided, default to "2023-12-01-preview"
    api_version = service_config.get("api_version", "2024-12-01-preview")

    if not endpoint or not api_key or not api_version:
        raise ValueError(f"Missing configuration for {service_type}")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        api_key=api_key
    )
    deployment_model = service_config.get("model")
    return client, deployment_model, endpoint, api_version

def test_llm_connection(client, model, endpoint, version):
    """
    Test the LLM (inference) connection using the AzureOpenAI client.
    """
    print("=== LLM (Inference) Connection Test ===")
    print(f"Model (deployment): {model}")
    print(f"Using API base: {endpoint}")
    print(f"Using API version: {version}")

    try:
        # In the new interface, provide the parameter 'model'
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the weather like today in Sweden Stockholm?"}
            ]
        )
        print("LLM connection test succeeded. Response:")
        print(response.choices[0].message.content)
    except Exception as e:
        print("LLM connection test failed with error:")
        print(e)

def test_embedding_connection(client, model, endpoint, version):
    """
    Test the embedding connection using the AzureOpenAI client.
    """
    print("\n=== Embedding Connection Test ===")
    print(f"Model (deployment): {model}")
    print(f"Using API base: {endpoint}")
    print(f"Using API version: {version}")

    try:
        response = client.embeddings.create(
            model=model,
            input="Hello, world!"
        )
        print("Embedding connection test succeeded. Response:")
        # Use attribute access: response.data is a list of objects with an 'embedding' attribute.
        print(response.data[0].embedding[:5])  # Print first 5 dimensions for brevity
    except Exception as e:
        print("Embedding connection test failed with error:")
        print(e)

if __name__ == "__main__":
    config = load_config()

    # Test LLM Connection
    try:
        llm_client, llm_model, llm_endpoint, llm_version = initialize_client(config, "llm")
        test_llm_connection(llm_client, llm_model, llm_endpoint, llm_version)
    except ValueError as e:
        print(f"LLM configuration error: {e}")

    # Test Embedding Connection
    try:
        emb_client, emb_model, emb_endpoint, emb_version = initialize_client(config, "embedding")
        test_embedding_connection(emb_client, emb_model, emb_endpoint, emb_version)
    except ValueError as e:
        print(f"Embedding configuration error: {e}")
