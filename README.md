# MyDeepSearcher
<div align="center">
  
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

# MyDeepSearcher evolves from deepsearcher opensource framework from zilliz
Owner: hanlian lyu

It support reasoning LLMs (OpenAI o1, o3-mini, DeepSeek, Grok 3, Claude 3.7 Sonnet, QwQ, etc.) and Vector Databases (Milvus, Zilliz Cloud etc.) to perform search, evaluation, and reasoning based on private data and internet data, providing highly accurate answer and comprehensive report. 
This project is most suitable for enterprise knowledge management, intelligent Q&A systems, and information retrieval scenarios.

# Features

- **Private Data Search**: Maximizes the utilization of enterprise internal data while ensuring data security. When necessary, it can integrate online content for more accurate answers.
- **Vector Database Management**: Supports Milvus and other vector databases, with data partitioning for efficient retrieval.
- **Flexible Embedding Options**: Compatible with multiple embedding models for optimal selection.
- **Multiple LLM Support**: Supports DeepSeek, OpenAI, and other large models for intelligent Q&A and content generation.
- **Document Loader**: Supports all common local files loading, with web crawling capabilities under development.
- **History Context Support**: Support multiround conversations and contexts transition.
- **Frontend and Thinking transparency**: User can see lively the Agent thinking, and engage in different levels of thinking depth.
- **Improved deepsearch speed, accuracy by 100%**: Use latest embedding and indexing technique, and parallel/async compute.


# 📖 Quick Start

### Clone the repository
git clone https://github.com/hanlianlu/mydeepsearcher.git

### MAKE SURE the python version is >= 3.12.3 for Mac/Linux, for WINDOWS == 3.11.9 .
### Recommended: Create a Python virtual environment
```
cd deep-searcher
python3 -m venv .venv
source .venv/bin/activate

### Install dependencies, this could lead to non-prod runnable, remember to remove related info in requirements when integrating to prod.
pip install -e .
```

### -----------------------------------DEVELOPMENT USAGE-----------------------

Prepare your `KEYS, END_POINTS, etc.` by creating '.env' file according to 'template_env.md'

#### Install Milvus Standalone version and run:
```sudo docker compose up -d
sudo docker compose down (close)
sudo docker compose ps (check)
```
#### Run App for User usage:
```
cd to deep_searcher_chat directory:
streamlit run prod_app.py
```


### ------------------------------------PRODUCTION USAGE------------------------


#### Production usage on VM (optional):
Ensure the VM’s firewall (e.g., UFW) and Azure Network Security Group allow traffic on port `8501, 19530, 9091, 8000`

#### Production usage on GIthub Actions workflow-file configuration (optional):
• Ensure Github Action workflow-file has: `[runs-on: ubuntu-latest]` IF under enterprise network, check with ADMIN for dedicated `runs-on`

#### Production usage on Azure Web App Service:
• Add Environment variables according to `template_env.md`
• Add SSH_PRIVATE_KEY (optional)
• Enable identity to system managed identity
• Set configuration start command with:[bash startup.sh]
