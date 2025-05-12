import pandas as pd
import sys
from dotenv import load_dotenv
import logging
import tkinter as tk
from tkinter import filedialog
from deepsearcher import configuration
from deepsearcher.online_query import query
from deepsearcher.configuration import Configuration, init_config
import ast
from deepsearcher.llm.azure_openai import AzureOpenAI
import os

# Function to evaluate LLM answer against golden-truth answer
def evaluate_answer(question: str, golden_answer: str, llm_answer: str, azure_openai_instance):
    """
    Uses Azure OpenAI to rate how good the LLM answer is compared to the golden-truth answer.
    Returns a score between 1 and 100.
    """
    # Print the lengths of the golden answer and LLM answer
    print(f"Golden Answer Length: {len(golden_answer)} characters\n ")
    print(f"LLM Answer Length: {len(llm_answer)} characters\n ")
    
    EVAL_PROMPT = """
    Given the question: {question}
    The correct answer is: {golden_answer}
    The model's answer is: {llm_answer}

    Respond ONLY with a single number with double decimal precision between 1 and 100 representing the matching quality, among which 100 being the best, 60 for minimum acceptable.
    """
    EVAL_PROMPT_INPUT = EVAL_PROMPT.format(question=question, golden_answer=golden_answer, llm_answer=llm_answer)
    print(f"Prompt Length: {len(EVAL_PROMPT_INPUT)} characters")
    
    chat_response = azure_openai_instance.chat(
        [{"role": "user", "content": EVAL_PROMPT_INPUT}]
    )
    
    try:
        score = ast.literal_eval(chat_response.content.strip())
        if isinstance(score, (int, float)):
            return min(max(round(score, 2), 1), 100)  # Clamp between 1 and 100
    except (ValueError, SyntaxError):
        logging.warning(f"Failed to parse response: {chat_response.content}")
        return 0

# Function to select a file using Tkinter dialog
def select_file(title):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(1)
    return file_path

# Main function to process the Excel file
def process_qa_pairs(azure_openai_instance, input_file=None, intermediate_file=None, output_file='output.xlsx'):
    if intermediate_file:
        df = pd.read_excel(intermediate_file)
        print(f"Using intermediate file: {intermediate_file}")
    else:
        df = pd.read_excel(input_file)
        print(f"Using input file: {input_file}")
        
        question_col = next(col for col in df.columns if 'question' in col.lower() or 'q' in col.lower())
        answer_col = next(col for col in df.columns if 'answer' in col.lower() or 'a' in col.lower())
        
        df = df.dropna(subset=[question_col])
        df = df[df[question_col].str.strip() != '']
        df[question_col] = df[question_col].str.strip()
        
        print("Generating LLM answers...")
        df['LLM-Answer'] = df[question_col].apply(lambda q: query(str(q), max_iter=2)[0])
        
        intermediate_output = 'query_results.xlsx'
        df.to_excel(intermediate_output, index=False)
        print(f"Intermediate query results saved to {intermediate_output}")
    
    question_col = next(col for col in df.columns if 'question' in col.lower() or 'q' in col.lower())
    answer_col = next(col for col in df.columns if 'answer' in col.lower() or 'a' in col.lower())
    llm_answer_col = next(col for col in df.columns if 'llm-answer' in col.lower())
    
    print("Evaluating LLM answers...")
    df['LLM-Score'] = df.apply(
        lambda row: evaluate_answer(
            row[question_col],
            row[answer_col],
            row[llm_answer_col],
            azure_openai_instance
        ),
        axis=1
    )
    
    average_score = df['LLM-Score'].mean(skipna=True)
    print(f"Average LLM Score: {average_score:.2f}")
    
    print(f"Writing complete results to {output_file}...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Detailed Results', index=False)
        pd.DataFrame({'Average Score': [average_score]}).to_excel(writer, sheet_name='Total Score', index=False)
    print("Processing complete!")

# Main execution
def main():
    config = Configuration()
    init_config(config=config)
    azure_openai_instance = configuration.backupllm

    # Check for --eval_only flag
    if len(sys.argv) > 1 and sys.argv[1] == "--eval_only":
        intermediate_file = select_file("Select Intermediate Excel File")
        process_qa_pairs(azure_openai_instance, intermediate_file=intermediate_file)
    else:
        input_file = select_file("Select Input Excel File")
        process_qa_pairs(azure_openai_instance, input_file=input_file)

if __name__ == '__main__':
    main()