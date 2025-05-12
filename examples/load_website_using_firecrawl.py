import logging
import os
from deepsearcher.offline_loading import load_from_website
from deepsearcher.online_query import query
from deepsearcher.configuration import Configuration, init_config

# Suppress unnecessary logging from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    # Step 1: Initialize configuration
    config = Configuration()


    # Apply the configuration
    init_config(config)

    # Step 2: Load data from a website into Milvus
    website_url = "https://en.wikipedia.org/wiki/Northvolt"  # Replace with your target website
    collection_name = "NorthvoltCrawl"
    collection_description = "Wiki Northvolt Documents"

    # crawl a single webpage
    load_from_website(urls=website_url, collection_name=collection_name, collection_description=collection_description)
    # only applicable if using Firecrawl: deepsearcher can crawl multiple webpages, by setting maxDepth, limit, allow_backward_links
    # load_from_website(urls=website_url, maxDepth=2, limit=20, allow_backward_links=True, collection_name=collection_name, collection_description=collection_description)

    # Step 3: Query the loaded data
    question = "Tell me about Northvolt especially its recent situation."  # Replace with your actual question
    result = query(question, max_iter= 1)
        


if __name__ == "__main__":
    main()
