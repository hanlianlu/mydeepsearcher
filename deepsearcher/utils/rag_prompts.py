# deepsearcher/utils/rag_prompts.py
__all__ = [
    "SUB_QUERY_PROMPT",
    "RERANK_PROMPT",
    "REFLECT_PROMPT",
    "SUMMARY_PROMPT",
]

# Prompts (unchanged from original deep searcher agent)-----------------------------------------------------------
SUB_QUERY_PROMPT = """To comprehensively answer the original question, follow these instructions:

1. Break down the original question into up to four sub-questions. Return as list of str.
2. Identify and interpret any acronyms or abbreviations present in the original question. If the meaning is uncertain, provide your best possible interpretation, you may form sub-questions to verify your interpretation.
3. If the history context is of relevance, put the relevant part into consideration altogether when you breakdown the original question.
4. If this is a very simple question and no decomposition is necessary, then just keep the original question in the python list.

Return your final result as a Python list of strings, with each sub-question clearly formulated and self-contained.

Original Question: {original_query}
History Context:   {history_context} 

<EXAMPLE>
Example input:
Original Question: "Explain deep learning"
History Context :"User: I'm new to Machine Learning, please suggest me a good starting course. AI: I recommend you take Machine Learning course from Andrew Ng as starter."

Example output:
[
    "What is deep learning?",
    "What is the difference between deep learning and machine learning?",
    "What is the history of deep learning?",
    "Does Andrew Ng provide a deep learning course, and is it also worthy for a beginner?"
]
</EXAMPLE>

Provide your response in a python list of str format:
"""

RERANK_PROMPT = """Based on the original query, sub-queries, and the retrieved chunks, determine for each chunk whether it is helpful in answering any of the queries and assign a relevance score between 0.00 and 1.00. 
Ensure scores are floats with double precision between 0.00 and 1.00.
When scoring relevance, apply a small positive bias to the original query so it edges out each secondary sub-query, but keep the weighing difference subtle rather than dominant.

Original Query: {original_query}
Sub-Queries: {sub_queries}
Retrieved Chunks: {retrieved_chunk}

Respond with a JSON list of lists, where each inner list corresponds to a chunk in the order provided and ONLY has the format ["YES"/"NO", score], e.g., [["YES", 0.91], ["NO", 0.22]] for two chunks. 
Ensure strings are quoted.
"""

REFLECT_PROMPT = """Determine whether additional search queries are needed based on the original query, previous sub queries, and the retrieved document chunks. If further research is required, provide a Python list of up to 3 search queries. If no further research is required, return an empty list.

Original Query: {question}

Previous Sub Queries: {mini_questions}

Retrieved Chunks: {chunk_str}

Respond exclusively in valid List of str format without any other text."""

SUMMARY_PROMPT = """
You are an AI content analysis expert, excellent at summarizing, analyzing content and providing insights and proposals. 
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
You may consider history contexts from user's previous conversation for reference.
Please evaluate and synthesize information from ALL of these chunks and docs to form your final answer to the user's query.
If the original query is simple or trivial, you should just answer concisely.

Original Query: {question}

Previous Sub Queries: {mini_questions}

Related Chunks: 
{mini_chunk_str}

History Contexts:
{history_context}
"""
# Prompts (unchanged from original deep searcher agent)-----------------------------------------------------------
