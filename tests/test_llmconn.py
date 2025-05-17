import os
import yaml
from dotenv import load_dotenv
from deepsearcher.llm.azure_openai import AzureOpenAI
from deepsearcher.llm.base import ChatResponse
import asyncio

# Load environment variables from .env file
load_dotenv()

# Load configuration from config.yaml
config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


# Extract LLM configuration from provide_settings.llm.config
model_config = config.get("provide_settings", {}).get("llm", {}).get("config", {})

# Override config with environment variables if they exist
azure_endpoint = model_config.get("azure_endpoint")
api_key = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = model_config.get("api_version")
model = model_config.get("model", "o3-mini")  # Default to "o3-mini" if not specified

# Print configuration for debugging
print("Model:", model)
print("AZURE_OPENAI_ENDPOINT:", azure_endpoint)
print("AZURE_OPENAI_KEY:", api_key)
print("AZURE_OPENAI_API_VERSION:", api_version)

# Sample conversation messages for testing
test_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
]

def test_sync_chat():
    """Test the synchronous chat method of AzureOpenAI."""
    try:
        # Initialize AzureOpenAI with loaded configuration
        llm = AzureOpenAI(
            model=model,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        # Call the synchronous chat method
        response: ChatResponse = llm.chat(test_messages)
        # Validate the response
        print("Synchronous Chat Response:", response.content)
        assert response.content is not None and len(response.content) > 0, "Response content is empty"
        assert response.total_tokens > 0, "Total tokens should be greater than 0"
        print("Synchronous chat test passed!")
    except Exception as e:
        print("Synchronous chat test failed:", e)

async def test_async_chat():
    """Test the asynchronous chat method of AzureOpenAI."""
    try:
        # Initialize AzureOpenAI with loaded configuration
        llm = AzureOpenAI(
            model=model,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        # Call the asynchronous chat method
        response: ChatResponse = await llm.chat_async(test_messages)
        # Validate the response
        print("Asynchronous Chat Response:", response.content)
        assert response.content is not None and len(response.content) > 0, "Response content is empty"
        assert response.total_tokens > 0, "Total tokens should be greater than 0"
        print("Asynchronous chat test passed!")
    except Exception as e:
        print("Asynchronous chat test failed:", e)

if __name__ == "__main__":
    # Run synchronous test
    print("Running synchronous chat test...")
    test_sync_chat()

    # Run asynchronous test
    print("\nRunning asynchronous chat test...")
    asyncio.run(test_async_chat())