from deepsearcher.configuration import Configuration, init_config
from deepsearcher import configuration
from deepsearcher.online_query import query, naive_rag_query

from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate
from datasets import Dataset

def fetch_answer_and_context(question: str):
    """
    Query the LLM with Milvus-enabled RAG and get answer + context.
    """
    print(f"\n Question: {question}")
    answer, context_docs = naive_rag_query(question)

    # Extract just the textual content of context docs
    context = ["\n".join(doc["content"] for doc in context_docs)]
    print(f"LLM Answer: {answer}")
    return answer, context

def run_ragas_evaluation(question: str, ground_truth: str):
    """
    Run the RAGAS evaluation on a single Q&A pair using Milvus-retrieved context.
    """
    answer, context = fetch_answer_and_context(question)

    dataset = Dataset.from_dict({
        "question": [question],
        "contexts": context,
        "ground_truth": [ground_truth],
        "answer": [answer],
    })

    # Evaluate using key RAGAS metrics
    results = evaluate(dataset, metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
    ])

    print("\n RAGAS Evaluation Results:")
    for metric, score in results.items():
        print(f"- {metric}: {score:.3f}")

def main():
    # Initialize config and Milvus connection
    config = Configuration()
    init_config(config=config)

    # This will hook into the Milvus vector store
    vector_db = configuration.vector_db
    print(f"Connected to Vector DB (Milvus): {type(vector_db).__name__}")

    # Sample query and ground truth
    sample_question = "What is the capital of France?"
    golden_answer = "Paris is the capital of France."

    run_ragas_evaluation(sample_question, golden_answer)

if __name__ == "__main__":
    main()