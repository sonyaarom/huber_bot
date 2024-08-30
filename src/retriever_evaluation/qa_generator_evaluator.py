import os
import sys
import json
import logging
import time
import random
import re
import argparse
from typing import List, Tuple
import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from tqdm import tqdm
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

cwd = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(cwd, '../assets')))
sys.path.insert(0, os.path.abspath(os.path.join(cwd, '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(cwd, '../models')))
sys.path.insert(0, os.path.abspath(os.path.join(cwd, '../')))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize models
embed_model = HuggingFaceEmbeddings()
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="../../models/Meta-Llama-3-8B-Instruct.Q3_K_L.gguf",
    temperature=0.5,
    max_tokens=512,
    top_p=0.95,
    top_k=50,
    callback_manager=callback_manager,
    verbose=True,
    n_ctx=8192,
    repeat_penalty=1.2,
    n_gpu_layers=1,
)

def extract_qa_pair(text: str) -> Tuple[str, str]:
    qa_pattern = re.compile(r'(?:Question:|Q:)\s*(.*?)\s*(?:Answer:|A:)\s*(.*)', re.DOTALL | re.IGNORECASE)
    match = qa_pattern.search(text)

    if match:
        return match.group(1).strip(), match.group(2).strip()

    question_pattern = re.compile(r'([^.!?]+\?)')
    questions = question_pattern.findall(text)

    if questions:
        question = questions[0].strip()
        answer = text[text.index(question) + len(question):].strip()
        return question, answer

    return None, None

def generate_single_question(text: str, max_retries: int = 3) -> List[Tuple[str, str]]:
    max_text_length = 300
    truncated_text = ' '.join(text.split()[:max_text_length])

    template = """Text: {text}

Generate 1 specific question and answer based on the text above. Follow these rules:

1. Use specific terms, names, and figures from the text in your question.
2. The question must be directly related to the text content.
3. The answer should be comprehensive and use information from the text.
4. Use only this format, with no additional text:

Q: [Specific question]
A: [Detailed answer]"""

    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(text=truncated_text)

    for attempt in range(max_retries):
        try:
            response = llm.invoke(formatted_prompt)
            question, answer = extract_qa_pair(response)

            if question and answer:
                return [(question, answer)]
            else:
                logging.warning(f"Attempt {attempt + 1}: Failed to extract Q&A. Retrying...")
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}: Error in generate_single_question: {str(e)}")

    logging.error(f"Failed to generate Q&A after {max_retries} attempts.")
    return []

def generate_qa_pairs(df: pd.DataFrame, max_questions: int = 5) -> List[Tuple[str, str, str]]:
    all_qa_pairs = []
    processed_ids = set()
    df = df.sample(frac=1).reset_index(drop=True)

    for _, row in df.iterrows():
        if len(all_qa_pairs) >= max_questions:
            break

        if row['id'] in processed_ids:
            continue

        logging.info(f"Processing row with ID: {row['id']}")
        qa_pairs = generate_single_question(row['text'])

        if qa_pairs:
            for question, answer in qa_pairs:
                all_qa_pairs.append((row['id'], question, answer))
                logging.info(f"Added QA pair for ID {row['id']}: Q: {question[:50]}... A: {answer[:50]}...")
                if len(all_qa_pairs) >= max_questions:
                    break
            processed_ids.add(row['id'])
        else:
            logging.warning(f"No QA pairs generated for row with ID: {row['id']}")

    logging.info(f"Total QA pairs generated: {len(all_qa_pairs)}")
    return all_qa_pairs

def clean_evaluated_questions(input_df: pd.DataFrame) -> pd.DataFrame:
    evaluated_df = input_df.copy()

    def clean_column(text: str) -> str:
        replacements = {
            'Question\n\n': '', 'Question\n': '', 'Question:': '',
            'Answer\n\n': '', 'Answer\n': '', 'Answer:': '',
            'Q:': '', 'A:': ''
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        for delimiter in ['\n\n**', '\n\n', '\n']:
            if delimiter in text:
                text = text.split(delimiter)[0]

        return text.strip()

    evaluated_df['question'] = evaluated_df['question'].apply(clean_column)
    evaluated_df['answer'] = evaluated_df['answer'].apply(clean_column)

    return evaluated_df

def is_non_specific_question(question):
    patterns = [
        r"^what is this about\??$",
        r"^what is the main topic\??$",
        r"^what (is|are) the main point(s)?\??$",
        r"^can you summarize this\??$",
        r"^what (is|are) the key takeaway(s)?\??$",
        r"^what is the main (point|idea|topic) of this (article|paper)\??$",
        r"^what information does the text provide\??$",
        r"^what is the main point of this paper\??$",
        r"^[Your specific question here]\??$",
        r"^what (does|do) the (model|paper|study) (propose|suggest|conclude)\??$",
        r"^how does the (model|paper|study) improve (on|over) existing (models|methods)\??$"
    ]

    question = question.lower()
    return any(re.match(pattern, question) for pattern in patterns)

@retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(20), retry=retry_if_exception_type((openai.RateLimitError, openai.APIError)))
def evaluate_question(question: str, client: openai.Client) -> dict:
    if is_non_specific_question(question):
        return {
            "Specificity": 1,
            "Realism": 1,
            "Clarity": 1
        }

    evaluation_prompt = f"""
    Evaluate the following question based on three criteria using a scale of 1 to 5:
    1. **Specificity**: Does the question clearly identify a specific paper, model, or study, or specific topic, avoiding general references like 'the model' or 'the paper'? The question should mention specific details that clearly delineate which content is being queried, rather than asking broadly.
    2. **Realism**: Is the question realistic and aligned with what students might genuinely ask in an academic setting? The question should reflect practical and common-sense inquiries likely to arise during study or review.
    3. **Clarity**: Is the question clearly formulated, avoiding ambiguous language or phrasing that could confuse students? The question should be easy to understand and free from vague terms or complex structures.
    Rate each criterion from 1 to 5, where:
    1 - Very Poor
    2 - Poor
    3 - Fair
    4 - Good
    5 - Excellent
    Question: "{question}"
    Please provide a rating for each criterion along with a brief explanation.
    Specificity:
    Realism:
    Clarity:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that evaluates questions based on specificity, realism, and clarity."},
                {"role": "user", "content": evaluation_prompt}
            ],
            max_tokens=150,
            temperature=0.5,
        )
        response_text = response.choices[0].message.content.strip()
    except openai.RateLimitError:
        logging.warning("Rate limit reached. Retrying...")
        raise
    except openai.AuthenticationError:
        logging.error("Authentication failed. Please check your API key.")
        raise
    except Exception as e:
        logging.error(f"Error in API call: {e}")
        return None

    scores = {
        "Specificity": None,
        "Realism": None,
        "Clarity": None
    }
    for line in response_text.splitlines():
        for criterion in scores.keys():
            if f"{criterion}:" in line:
                try:
                    scores[criterion] = int(line.split(":")[1].strip().split()[0])
                except (ValueError, IndexError):
                    logging.error(f"Error parsing score for {criterion}")

    return scores

def process_questions(df: pd.DataFrame, client: openai.Client, batch_size: int = 1) -> pd.DataFrame:
    evaluated_questions_df = df.copy()
    evaluated_questions_df['Specificity'] = None
    evaluated_questions_df['Realism'] = None
    evaluated_questions_df['Clarity'] = None
    evaluated_questions_df['Average Score'] = None

    for i in tqdm(range(0, len(evaluated_questions_df), batch_size), desc="Processing questions"):
        batch = evaluated_questions_df.iloc[i:i+batch_size]
        for index, row in batch.iterrows():
            scores = evaluate_question(row['question'], client)
            if scores:
                valid_scores = [score for score in scores.values() if score is not None]
                for criterion, score in scores.items():
                    evaluated_questions_df.at[index, criterion] = score
                if valid_scores:
                    average_score = sum(valid_scores) / len(valid_scores)
                    evaluated_questions_df.at[index, 'Average Score'] = average_score

            time.sleep(random.uniform(5, 15))

    return evaluated_questions_df


def main(input_file: str, output_file: str, max_questions: int = 1, sample_size: int = 0):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

    # Set up OpenAI client
    client = openai.Client(api_key=api_key)

    # Load data
    data_full = pd.read_csv(input_file, index_col=0)
    data_full = data_full.dropna(subset=['text'])

    # Apply sampling if specified
    if sample_size > 0:
        data_full = data_full.sample(n=min(sample_size, len(data_full)), random_state=42)
    
    logging.info(f"Processing {len(data_full)} rows")

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(data_full, max_questions=max_questions)
    qa_df = pd.DataFrame(qa_pairs, columns=['id', 'question', 'answer'])
    qa_df = clean_evaluated_questions(qa_df)

    # Evaluate questions
    evaluated_questions_df = process_questions(qa_df, client)

    # Save results
    evaluated_questions_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and evaluate question-answer pairs from text data.")
    parser.add_argument("--input", type=str, default="../../assets/csv/data_full.csv", help="Path to the input CSV file")
    parser.add_argument("--output", type=str, default="../../assets/csv/evaluated_questions_with_scores.csv", help="Path to the output CSV file")
    parser.add_argument("--max_questions", type=int, default=5, help="Maximum number of questions to generate")
    parser.add_argument("--sample", type=int, default=0, help="Number of rows to sample (0 means no sampling)")
    
    args = parser.parse_args()
    
    main(args.input, args.output, args.max_questions, args.sample)