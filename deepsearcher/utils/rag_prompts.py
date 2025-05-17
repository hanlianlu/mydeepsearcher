# deepsearcher/utils/rag_prompts.py
# ---------------------------------------------------------------------------
# Prompt templates for various RAG-based agents, including Chain-of-RAG monolith
# ---------------------------------------------------------------------------

__all__ = [
    # DeepSearcher core prompts
    "SUB_QUERY_PROMPT",
    "RERANK_PROMPT",
    "REFLECT_PROMPT",
    "SUMMARY_PROMPT",
    "COLLECTION_ROUTE_PROMPT",
    "RAG_ROUTER_PROMPT",
    "CONFIDENCE_PROMPT",
    "COR_INTERMEDIATE_PROMPT",
    "COR_REFLECT_PROMPT",
    # Chain-of-RAG monolith prompts (prefixed to avoid naming conflicts)
    "MONO_FOLLOWUP_PROMPT",
    "MONO_INTERMEDIATE_PROMPT",
    "MONO_FINAL_PROMPT",
    "MONO_CONFIDENCE_PROMPT",
    # Micro-flow aliases (explicit mapping to monolith prompts)
    "CHAINOF_RAG_PLAN_PROMPT",
    "CHAINOF_RAG_REFINE_PROMPT",
    "CHAINOF_RAG_SUMMARY_PROMPT",
]

# ──────────────────────────────────────────────────────────────────────────
# DeepSearcher core prompts (unchanged)
# ──────────────────────────────────────────────────────────────────────────
SUB_QUERY_PROMPT = """To comprehensively answer the original question, follow these instructions:

1. Break down the original question into up to four sub-questions. Return as list of str.
2. Identify and interpret any acronyms or abbreviations present in the original question. If the meaning is uncertain, provide your best possible interpretation, you may form sub-questions to verify your interpretation.
3. If the history context is of relevance, put the relevant part into consideration altogether when you breakdown the original question.
4. If this is a very simple question and no decomposition is necessary, then just keep the original question in the Python list.

Return your final result as a Python list of strings, with each sub-question clearly formulated and self-contained.

Original Question: {original_query}
History Context: {history_context}

<EXAMPLE>
Example input:
Original Question: "Explain deep learning"
History Context: "User: I'm new to Machine Learning, please suggest me a good starting course. AI: I recommend you take Machine Learning course from Andrew Ng as starter."

Example output:
[
    "What is deep learning?",
    "What is the difference between deep learning and machine learning?",
    "What is the history of deep learning?",
    "Does Andrew Ng provide a deep learning course, and is it also worthy for a beginner?"
]
</EXAMPLE>

Provide your response in a Python list of str format:
"""

RERANK_PROMPT = """Based on the original query, sub-queries, and the retrieved chunks, determine for each chunk whether it is helpful in answering any of the queries and assign a relevance score between 0.00 and 1.00.
Ensure scores are floats with double precision between 0.00 and 1.00.
When scoring relevance, apply a small positive bias to the original query so it edges out each secondary sub-query, but keep the weighing difference subtle rather than dominant.

Original Query: {original_query}
Sub-Queries: {sub_queries}
Retrieved Chunks: {retrieved_chunk}

Respond with a JSON list of lists, where each inner list corresponds to a chunk in the order provided and ONLY has the format ["YES", score] or ["NO", score], e.g., [["YES", 0.91], ["NO", 0.22]]. Ensure strings are quoted.
"""

REFLECT_PROMPT = """Determine whether additional search queries are needed based on the original query, previous sub queries, and the retrieved document chunks. If further research is required, provide a Python list of up to 3 search queries. If no further research is required, return an empty list.

Original Query: {question}

Previous Sub Queries: {mini_questions}

Retrieved Chunks: {chunk_str}

Respond exclusively in valid List of str format without any other text.
"""

SUMMARY_PROMPT = """You are an AI content analysis expert, excellent at summarizing, analyzing content and providing insights and proposals.
Please produce your final answer **in Markdown format**, observing these rules:
1. **Headings**  
   - Use `#`, `##`, `###`, etc. to structure major sections.
2. **Bullet lists**  
   - Use `-` or `*` for unordered lists, and `1., 2., 3., …` for ordered lists.
3. **Tables**  
   - Any tabular data must be formatted as a pipe‑delimited Markdown table with a header row and separator line.
4. **Code snippets**  
   - Use triple backticks (```) for any code or example commands.
5. **Emphasis**  
   - Use `**bold**` or `*italic*` where appropriate.
6. **Links and images**  
   - If referencing URLs or images, use standard Markdown syntax: `[text](url)` or `![alt](url)`.
Please summarize a specific and detailed answer or report based on the previous queries and the retrieved document chunks. You may then provide professional proposals and insights more extensively.
Consider history contexts from user's previous conversation for reference.
Evaluate and synthesize information from ALL chunks and docs to form your final answer to the user's query.
If the original query is simple or trivial, answer concisely.

Original Query: {question}

Previous Sub Queries: {mini_questions}

Related Chunks:
{mini_chunk_str}

History Contexts:
{history_context}
"""

COLLECTION_ROUTE_PROMPT = """You are provided with:
  - a QUESTION (string)
  - a COLLECTION_INFO list of dicts, each with keys
      • collection_name
      • collection_description
  - the Current Iteration (“first” or “subsequent”)

Your task:
1. Identify which collection_name(s) are relevant to answer the QUESTION.
2. Strictly IGNORE any “Current Iteration” data unless it is “first.”

Rules:
- If Iteration is “first” AND you find NO relevant collections, you MUST return ALL available collection_name(s).
- Return ONLY a valid Python list of strings (the selected names), with NO extra commentary.

QUESTION: {question}

COLLECTION_INFO: {collection_info!r}

Current Iteration: {curr_iter_str}

Return your list below:
"""

RAG_ROUTER_PROMPT = """You are given:
  - A user QUESTION (string)
  - A numbered list of agent descriptions, each of the form “\[1\]: description…”

Your job is to pick **exactly one** agent by its **1-based index**, that would best handle the QUESTION.

Return just the index (e.g. “1” or “2”)—no other text.

QUESTION:
{query}

AGENTS:
{description_str}

Only reply with the single number (1, 2, or 3).
"""

CONFIDENCE_PROMPT = """Based on the original query, previous sub-queries, and the retrieved chunks, assess your confidence (as a number between 0 and 1) that you have enough information to answer the query comprehensively with profound insights.
Original Query: {original_query}
Previous Sub Queries: {sub_queries}
Retrieved Chunks: {chunk_str}
Respond ONLY with a single number with double decimal precision between 0.00 and 1.00.
"""

COR_INTERMEDIATE_PROMPT = """..."""

COR_REFLECT_PROMPT = """..."""

# ──────────────────────────────────────────────────────────────────────────
# Chain-of-RAG monolith prompts (renamed with MONO_ prefix)
# ──────────────────────────────────────────────────────────────────────────
MONO_FOLLOWUP_PROMPT = """
You are using a search tool to answer the main query by iteratively searching the database. Given the following intermediate queries and answers, identify what key information is still needed to fully answer the main query. Generate up to three follow-up questions that target these gaps. Each question should be distinct and aim to uncover a specific piece of information that has not been addressed yet. If fewer than three questions are needed, provide only those.

## Previous intermediate queries and answers
{intermediate_context}

## Main query to answer
{query}

Respond with a Python list of follow-up questions (e.g., ["question1", "question2"]). Do not explain yourself or output anything else.
"""

MONO_INTERMEDIATE_PROMPT = """
Given the following documents, generate an appropriate answer for the query. DO NOT hallucinate any information, only use the provided documents to generate the answer. Respond "No relevant information found" if the documents do not contain useful information.

## Documents
{retrieved_documents}

## Query
{sub_query}

Respond with a concise answer only, do not explain yourself or output anything else.
"""

MONO_FINAL_PROMPT = """
You are an AI content analysis expert. Based on the following retrieved documents, intermediate queries and answers, and history context, generate a final comprehensive answer for the main query. Your final answer must include two clearly demarcated sections:

Section 1: Factual Analysis
- Provide a structured synthesis strictly from documents and intermediate answers. Cite with [Document X] or [Intermediate Answer Y].

Section 2: AI Augmented Insights
- Add broader context or interpretation beyond the retrieved facts.

## Retrieved Documents:
{retrieved_documents}
## Intermediate Queries and Answers:
{intermediate_context}
## Main Query:
{query}
## History Context:
{history_context}

Respond in Markdown with the headings "Factual Analysis:" and "AI Augmented Insights:". Do not include extra text.
"""

MONO_CONFIDENCE_PROMPT = """
Given the following intermediate queries and answers, estimate your confidence (0.00 to 1.00) that you have enough information to answer the main query.

## Intermediate queries and answers
{intermediate_context}

## Main query
{query}

Respond with a single number with two decimal places between 0.00 and 1.00 only.
"""

# ──────────────────────────────────────────────────────────────────────────
# Micro-flow aliases mapping to monolith prompts
# ──────────────────────────────────────────────────────────────────────────
CHAINOF_RAG_PLAN_PROMPT = MONO_FOLLOWUP_PROMPT
CHAINOF_RAG_REFINE_PROMPT = MONO_INTERMEDIATE_PROMPT
CHAINOF_RAG_SUMMARY_PROMPT = MONO_FINAL_PROMPT
