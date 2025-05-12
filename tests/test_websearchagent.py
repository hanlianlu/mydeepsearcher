import asyncio
import logging
from deepsearcher.configuration import Configuration, init_config
from deepsearcher import configuration

# Test function for WebSearchAgent as a whole
async def test_web_search_agent():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize configuration as specified
    config = Configuration()
    init_config(config=config)
    
    webagent = configuration.web_search_agent
    # Instantiate WebSearchAgent with required components from config

    # Sample query
    query = "machine learning advancements"
    logger.info(f"Testing WebSearchAgent with query: '{query}'")

    # Test the agent's retrieve method
    try:
        results, tokens, metadata = await webagent.retrieve(query=query)
        
        # Log the overall results
        logger.info(f"Retrieved {len(results)} documents")
        for i, res in enumerate(results, 1):
            logger.info(f"Document {i}:")
            logger.info(f"  Reference: {res.reference}")
            logger.info(f"  Score: {res.score:.2f}")
            logger.info(f"  Text: {res.text[:50]}...")
        logger.info(f"Estimated tokens used: {tokens}")
        logger.info(f"Metadata: {metadata}")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(test_web_search_agent())