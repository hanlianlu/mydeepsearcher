# Some test dataset and evaluation method are ref from https://github.com/OSU-NLP-Group/HippoRAG/tree/main/data , many thanks

################################################################################
# Note: This evaluation script will cost a lot of LLM token usage, please make sure you have enough token budget.
################################################################################
import argparse
import ast
import json
import logging
import os
import time
import warnings
from collections import defaultdict
from typing import List, Tuple

import pandas as pd

from deepsearcher.configuration import Configuration, init_config
from deepsearcher.offline_loading import load_from_local_files
from deepsearcher.online_query import naive_retrieve, retrieve
from deepsearcher import configuration  # Added for accessing vector_db

httpx_logger = logging.getLogger("httpx")  # disable openai's logger output
httpx_logger.setLevel(logging.WARNING)

warnings.simplefilter(action="ignore", category=FutureWarning)  # disable warning output

current_dir = os.path.dirname(os.path.abspath(__file__))

k_list = [2, 5]

def _deepsearch_retrieve_titles(
    question: str,
    retry_num: int = 2,
    base_wait_time: int = 4,
    max_iter: int = 3,
) -> Tuple[List[str], int, bool]:
    """
    Retrieve document titles using DeepSearcher with retry mechanism.

    Args:
        question (str): The query question.
        retry_num (int, optional): Number of retry attempts. Defaults to 4.
        base_wait_time (int, optional): Base wait time between retries in seconds. Defaults to 4.
        max_iter (int, optional): Maximum number of iterations for retrieval. Defaults to 3.

    Returns:
        Tuple[List[str], int, bool]: A tuple containing:
            - List of retrieved document titles
            - Number of tokens consumed
            - Boolean indicating whether the retrieval failed
    """
    retrieved_results = []
    consume_tokens = 0
    for i in range(retry_num):
        try:
            retrieved_results, _, consume_tokens = retrieve(question, max_iter=max_iter, collections_names=["deepsearcher"] )
            break
        except Exception:
            wait_time = base_wait_time * (2**i)
            print(f"Parse LLM's output failed, retry again after {wait_time} seconds...")
            time.sleep(wait_time)
    if retrieved_results:
        retrieved_titles = [
            retrieved_result.metadata["title"] for retrieved_result in retrieved_results
        ]
        fail = False
    else:
        print("Pipeline error, no retrieved results.")
        retrieved_titles = []
        fail = True
    return retrieved_titles, consume_tokens, fail

def _naive_retrieve_titles(question: str) -> List[str]:
    """
    Retrieve document titles using naive retrieval method.

    Args:
        question (str): The query question.

    Returns:
        List[str]: List of retrieved document titles.
    """
    retrieved_results = naive_retrieve(question, collections_names=["deepsearcher"])
    retrieved_titles = [
        retrieved_result.metadata["title"] for retrieved_result in retrieved_results
    ]
    return retrieved_titles

def _calcu_recall(sample, retrieved_titles, dataset) -> dict:
    """
    Calculate recall metrics for retrieved titles.

    Args:
        sample: The sample data containing ground truth information.
        retrieved_titles: List of retrieved document titles.
        dataset (str): The name of the dataset being evaluated.

    Returns:
        dict: Dictionary containing recall values at different k values.

    Raises:
        NotImplementedError: If the dataset is not supported.
    """
    if dataset in ["2wikimultihopqa"]:
        gold_passages = [item for item in sample["supporting_facts"]]
        gold_items = set([item[0] for item in gold_passages])
        retrieved_items = retrieved_titles
    else:
        raise NotImplementedError

    recall = dict()
    for k in k_list:
        recall[k] = round(
            sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items), 4
        )
    return recall

def _print_recall_line(recall: dict, pre_str="", post_str="\n"):
    """
    Print recall metrics in a formatted line.

    Args:
        recall (dict): Dictionary containing recall values at different k values.
        pre_str (str, optional): String to print before recall values. Defaults to "".
        post_str (str, optional): String to print after recall values. Defaults to "\n".
    """
    print(pre_str, end="")
    for k in k_list:
        print(f"R@{k}: {recall[k]:.3f} ", end="")
    print(post_str, end="")

def evaluate(
    dataset: str,
    output_root: str,
    pre_num: int = 10,
    max_iter: int = 3,
    skip_load=False, #RESET TO FALSE IF NEED TO DOWNLOAD NEW DATASET
    flag: str = "result",
):
    """
    Evaluate the retrieval performance on a dataset.

    Args:
        dataset (str): Name of the dataset to evaluate.
        output_root (str): Root directory for output files.
        pre_num (int, optional): Number of samples to evaluate. Defaults to 10.
        max_iter (int, optional): Maximum number of iterations for retrieval. Defaults to 3.
        skip_load (bool, optional): Whether to skip loading the dataset. Defaults to False.
        flag (str, optional): Flag for the evaluation run. Defaults to "result".
    """
    corpus_file = os.path.join(current_dir, f"../examples/data/{dataset}_corpus.json")
    if not skip_load:
        # set chunk size to a large number to avoid chunking, because the dataset was chunked already.
        load_from_local_files(
            corpus_file, force_new_collection=True, chunk_size=999999, chunk_overlap=0, collection_name="deepsearcher"
        )

    eval_output_subdir = os.path.join(output_root, flag)
    os.makedirs(eval_output_subdir, exist_ok=True)
    csv_file_path = os.path.join(eval_output_subdir, "details.csv")
    statistics_file_path = os.path.join(eval_output_subdir, "statistics.json")

    data_with_gt_file_path = os.path.join(current_dir, f"../examples/data/{dataset}.json")
    data_with_gt = json.load(open(data_with_gt_file_path, "r"))

    if not pre_num:
        pre_num = len(data_with_gt)

    pipeline_error_num = 0
    end_ind = min(pre_num, len(data_with_gt))

    start_ind = 0
    existing_df = pd.DataFrame()
    existing_statistics = defaultdict(dict)
    existing_token_usage = 0
    existing_error_num = 0
    existing_sample_num = 0
    if os.path.exists(csv_file_path):
        existing_df = pd.read_csv(csv_file_path)
        start_ind = len(existing_df)
        print(f"Loading results from {csv_file_path}, start_index = {start_ind}")

    if os.path.exists(statistics_file_path):
        existing_statistics = json.load(open(statistics_file_path, "r"))
        print(
            f"Loading statistics from {statistics_file_path}, will recalculate the statistics based on both new and existing results."
        )

        existing_token_usage = existing_statistics["deepsearcher"]["token_usage"]
        existing_error_num = existing_statistics["deepsearcher"].get("error_num", 0)
        existing_sample_num = existing_statistics["deepsearcher"].get("sample_num", 0)
    for sample_idx, sample in enumerate(data_with_gt[start_ind:end_ind]):
        global_idx = sample_idx + start_ind
        question = sample["question"]

        retrieved_titles, consume_tokens, fail = _deepsearch_retrieve_titles(
            question, max_iter=max_iter
        )
        retrieved_titles_naive = _naive_retrieve_titles(question)

        if fail:
            pipeline_error_num += 1
            print(
                f"Pipeline error, no retrieved results. Current pipeline_error_num = {pipeline_error_num}"
            )

        print(f"idx: {global_idx}: ")
        recall = _calcu_recall(sample, retrieved_titles, dataset)
        recall_naive = _calcu_recall(sample, retrieved_titles_naive, dataset)
        current_result = [
            {
                "idx": global_idx,
                "question": question,
                "recall": recall,
                "recall_naive": recall_naive,
                "gold_titles": [item[0] for item in sample["supporting_facts"]],
                "retrieved_titles": retrieved_titles,
                "retrieved_titles_naive": retrieved_titles_naive,
            }
        ]
        current_df = pd.DataFrame(current_result)
        existing_df = pd.concat([existing_df, current_df], ignore_index=True)
        existing_df.to_csv(csv_file_path, index=False)
        average_recall = dict()
        average_recall_naive = dict()
        for k in k_list:
            average_recall[k] = sum(
                [
                    ast.literal_eval(d).get(k) if isinstance(d, str) else d.get(k)
                    for d in existing_df["recall"]
                ]
            ) / len(existing_df)
            average_recall_naive[k] = sum(
                [
                    ast.literal_eval(d).get(k) if isinstance(d, str) else d.get(k)
                    for d in existing_df["recall_naive"]
                ]
            ) / len(existing_df)
        _print_recall_line(average_recall, pre_str="Average recall of DeepSearcher: ")
        _print_recall_line(average_recall_naive, pre_str="Average recall of naive RAG   : ")
        existing_token_usage += consume_tokens
        existing_error_num += 1 if fail else 0
        existing_sample_num += 1
        existing_statistics["deepsearcher"]["average_recall"] = average_recall
        existing_statistics["deepsearcher"]["token_usage"] = existing_token_usage
        existing_statistics["deepsearcher"]["error_num"] = existing_error_num
        existing_statistics["deepsearcher"]["sample_num"] = existing_sample_num
        existing_statistics["deepsearcher"]["token_usage_per_sample"] = (
            existing_token_usage / existing_sample_num
        )
        existing_statistics["naive_rag"]["average_recall"] = average_recall_naive
        json.dump(existing_statistics, open(statistics_file_path, "w"), indent=4)
        print("")
    print("Finish results to save.")

def main_eval():
    parser = argparse.ArgumentParser(prog="evaluate", description="Deep Searcher evaluation.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="2wikimultihopqa",
        help="Dataset name, default is `2wikimultihopqa`. More datasets will be supported in the future.",
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./eval_config.yaml",
        help="Configuration yaml file path, default is `./eval_config.yaml`",
    )
    parser.add_argument(
        "--pre_num",
        type=int,
        default=30,
        help="Number of samples to evaluate, default is 30",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=3,
        help="Max iterations of reflection. Default is 3. It will overwrite the one in config yaml file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_output",
        help="Output root directory, default is `./eval_output`",
    )
    parser.add_argument(
        "--skip_load",
        action="store_true",
        help="Whether to skip loading the dataset. Default it don't skip loading. If you want to skip loading, please set this flag.",
    )
    parser.add_argument(
        "--flag",
        type=str,
        default="result",
        help="Flag for evaluation, default is `result`",
    )

    args = parser.parse_args()

    config = Configuration(config_path=args.config_yaml)
    init_config(config=config)
    
    # Ensure only "deepsearcher" collection is used
    configuration.vector_db.default_collection = "deepsearcher"
    print("Set default collection to 'deepsearcher'")

    evaluate(
        dataset=args.dataset,
        output_root=args.output_dir,
        pre_num=args.pre_num,
        max_iter=args.max_iter,
        skip_load= args.skip_load,
        flag=args.flag,
    )

if __name__ == "__main__":
    main_eval()