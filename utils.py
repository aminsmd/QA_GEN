import os
import time
import json
import pandas as pd

# autogen and LLM-related imports
import autogen
from autogen import AssistantAgent, config_list_from_json
from autogen.token_count_utils import count_token

# pydantic models
from pydantic import BaseModel

# instructor / openai
import instructor
from openai import OpenAI

# For custom example
from types import SimpleNamespace
from typing import List

# Load API key from config.json
with open('config.json') as config_file:
    config = json.load(config_file)
    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]

# -------------------------------------------------------------------
# Pydantic Models for parsing responses
# -------------------------------------------------------------------
class ExtractedInfo_Base(BaseModel):
    question: str
    answer: str

class ExtractedInfo_Solution(BaseModel):
    question: str
    solution: str
    answer: str

class ExtractedInfo_COT(BaseModel):
    question: str
    reasoning: str
    answer: str

# -------------------------------------------------------------------
# Base Custom LLM Client (refactored to reduce redundancy)
# -------------------------------------------------------------------
class CustomLLMClient:
    """
    A custom LLM client that uses the instructor library (with an OpenAI backend)
    to parse JSON responses into ExtractedInfo objects.
    This base class handles all common functionality.
    Subclasses must set the class variable `response_model`.
    """
    response_model = None  # To be defined by subclasses

    def __init__(self, config, **kwargs):
        # Optional: add further configuration or logging here
        model="gpt-4o",
        print(f"CustomLLMClient config: {config}")

    def create(self, params):
        client = instructor.from_openai(
            OpenAI(base_url="https://api.openai.com/v1/", 
                   api_key=os.environ["OPENAI_API_KEY"]),
            mode=instructor.Mode.JSON
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=params["messages"],
            response_model=self.response_model,
        )
        # Construct an autogen-like response
        autogen_response = SimpleNamespace()
        autogen_response.choices = []
        autogen_response.model = "custom_llm_extractor"
        choice = SimpleNamespace()
        choice.message = SimpleNamespace()
        choice.message.content = response.model_dump_json()
        choice.message.function_call = None
        autogen_response.choices.append(choice)
        return autogen_response

    def message_retrieval(self, response):
        return [choice.message.content for choice in response.choices]

    def cost(self, response) -> float:
        response.cost = 0
        return 0

    @staticmethod
    def get_usage(response):
        return {}

# -------------------------------------------------------------------
# Specialized LLM Clients using the base client
# -------------------------------------------------------------------
class CustomLLMClient_Base(CustomLLMClient):
    response_model = ExtractedInfo_Base

class CustomLLMClient_Solution(CustomLLMClient):
    response_model = ExtractedInfo_Solution

class CustomLLMClient_COT(CustomLLMClient):
    response_model = ExtractedInfo_COT
    
    
# -------------------------------------------------------------------
# Model config to attach the custom client to the "custom_llm_extractor"
# -------------------------------------------------------------------
config_list = [
    {
        "model": "custom_llm_extractor",
        "model_client_cls": "CustomLLMClient",
    }
]

# -------------------------------------------------------------------
# Helper: Token counting
# -------------------------------------------------------------------
def calculate_total_tokens(messages, model="gpt-4o"):
    """
    Use autogen's count_token to measure usage across messages.
    """
    total_tokens = 0
    for message in messages:
        content = message.get("content", "")
        if content:
            tokens = count_token(content, model=model)
            total_tokens += tokens
    return total_tokens

# -------------------------------------------------------------------
# Prompts for Evaluations
# -------------------------------------------------------------------
CLARITY_SYSTEM_PROMPT = (
    "You are an experienced educator specializing in math for mid-school pupils. You are known for your high standards and "
    "strict grading, awarding full marks only when the work is truly exceptional. Your goal is to critically evaluate "
    "math questions written by a junior teacher, focusing on their clarity.\n\n"
    "For each question, follow these steps:\n"
    "1. Read the provided question and its answer carefully.\n"
    "2. Identify any weaknesses or ambiguities in the question—even minor ones.\n"
    "3. Decide on a clarity score based on the following scale:\n"
    "   (1) Not at all clear\n"
    "   (2) Mostly unclear\n"
    "   (3) Somewhat clear\n"
    "   (4) Mostly clear\n"
    "   (5) Very clear\n\n"
    "Before providing your final score (1–5), explain your reasoning step by step. Your explanation should detail how "
    "you arrived at the score based on the clarity of the question. Only award high scores if the question is truly exceptional.\n\n"
    "Remember to evaluate the question through the lens of a middle school student—that is, while an adult might find the "
    "question clear, assess whether it is appropriately challenging and understandable for middle school pupils."
)

RELEVANCE_SYSTEM_PROMPT = (
    "You are an experienced educator specializing in mathematics. Your task is to evaluate the relevance of a math question \n"
    "to a specific knowledge component (KC). The set of example problems are aligned with the target knowledge component \n"
    "Evaluate whether the question aligns with the "
    "target knowledge component and covers the same topic as the provided examples.\n\n"
    "Instructions:\n"
    "1. Read the provided question, target knowledge component (KC), and example problems carefully.\n"
    "2. Identify any gaps or misalignments between the question and the target KC or examples.\n"
    "3. Assign a relevance score on a scale from 1 to 5, where:\n"
    "   (1) Not at all relevant\n"
    "   (2) Mostly irrelevant\n"
    "   (3) Somewhat relevant\n"
    "   (4) Mostly relevant\n"
    "   (5) Highly relevant\n\n"
    "4. Provide your reasoning step-by-step, detailing how you arrived at your relevance score.\n\n"
    "Remember to evaluate the question through the lens of a middle school student—ensure that the question is relevant and "
    "comprehensible for that audience."
)

IMPORTANCE_SYSTEM_PROMPT = (
    "You are an experienced educator specializing in mathematics. Your task is to evaluate the importance of a math question "
    "in addressing a critical aspect of a specific knowledge component (KC). Evaluate whether the question targets a key "
    "concept and whether it effectively focuses on essential aspects of the KC.\n\n"
    "Instructions:\n"
    "1. Read the provided question and the target knowledge component (KC) carefully.\n"
    "2. Consider how well the question addresses critical elements of the KC.\n"
    "3. Assign an importance score on a scale from 1 to 5, where:\n"
    "   (1) Not at all important\n"
    "   (2) Slightly important\n"
    "   (3) Moderately important\n"
    "   (4) Very important\n"
    "   (5) Essential\n\n"
    "4. Provide your reasoning step-by-step, detailing how you arrived at your importance score.\n\n"
    "Remember to evaluate the question from the perspective of a middle school student, ensuring that it is both engaging "
    "and essential for their learning."
)

DIFFICULTY_MATCHING_SYSTEM_PROMPT = (
    "You are an experienced educator specializing in mathematics. Your task is to evaluate how well a generated math "
    "question matches a specified difficulty level for a given knowledge component (KC), based on a set of example problems. "
    "The specified difficulty level will be provided as one of the following: easy, medium, or hard. Your evaluation should "
    "determine whether the question’s complexity and cognitive demands align with the intended difficulty level. For example, "
    "an 'easy' question should test basic facts with minimal reasoning, a 'medium' question should require moderate reasoning, "
    "and a 'hard' question should demand advanced reasoning and deeper understanding.\n\n"
    "Instructions:\n"
    "1. Read the provided question, target knowledge component (KC), and examples carefully.\n"
    "2. Note the specified difficulty level (easy, medium, or hard).\n"
    "3. Evaluate the cognitive demands and complexity of the question by considering the following:\n"
    "   - For 'easy': Does the question test basic facts or simple concepts?\n"
    "   - For 'medium': Does the question require moderate reasoning or application of concepts?\n"
    "   - For 'hard': Does the question demand advanced reasoning, analysis, or synthesis?\n"
    "4. Compare the evaluated complexity of the question with what is expected for the specified difficulty level.\n"
    "5. Assign a difficulty matching score on a scale from 1 to 5, where:\n"
    "   (1) No alignment with the specified difficulty\n"
    "   (2) Slight alignment\n"
    "   (3) Moderate alignment\n"
    "   (4) Mostly aligned\n"
    "   (5) Perfectly aligned\n\n"
    "6. Provide your reasoning step-by-step, detailing how you arrived at your score.\n\n"
    "Remember to evaluate the question through the lens of a middle school student—consider whether its complexity "
    "is appropriate for that level."
)

ANSWERABILITY_SYSTEM_PROMPT = (
    "You are an experienced educator specializing in mathematics for middle school students. Your task is to evaluate whether "
    "a generated math question is answerable by a middle school student, taking into account both clarity and completeness of information. "
    "Assess if the question is stated clearly, includes all necessary details, and is pitched at an appropriate level for that audience.\n\n"
    "Instructions:\n"
    "1. Read the provided question and its answer carefully.\n"
    "2. Identify any ambiguities or missing information that might prevent a typical middle school student from understanding or answering the question.\n"
    "3. Evaluate whether the language and structure of the question are suitable for a middle school student.\n"
    "4. Based on your assessment, assign an answerability score on a scale from 1 to 5, where:\n"
    "   (1) Not answerable at all\n"
    "   (2) Barely answerable\n"
    "   (3) Somewhat answerable\n"
    "   (4) Mostly answerable\n"
    "   (5) Clearly answerable\n\n"
    "5. Provide your reasoning step-by-step, detailing how you arrived at your score.\n\n"
    "Remember to evaluate the question from the perspective of a middle school student—what may seem clear to an adult must be "
    "assessed for its understandability and answerability for a middle school pupil."
)

BLOOM_EVALUATION_SYSTEM_PROMPT = (
    "You are an experienced educator specializing in mathematics education. Your task is to evaluate whether a generated math "
    "question aligns with the intended cognitive challenge level as defined by Bloom's Taxonomy. Bloom's Taxonomy comprises six "
    "levels of learning objectives:\n\n"
    "1. Remembering: Recalling facts or basic concepts.\n"
    "2. Understanding: Comprehending the meaning of information and explaining it in your own words.\n"
    "3. Applying: Using information to solve problems in new situations.\n"
    "4. Analyzing: Breaking down information into parts and understanding its structure or relationships.\n"
    "5. Evaluating: Making judgments about the value or quality of information.\n"
    "6. Creating: Synthesizing information to produce new ideas or products.\n\n"
    "In our system, we define three difficulty levels for generated questions:\n"
    " - Easy: Questions should primarily test the first two Bloom levels (Remembering and Understanding).\n"
    " - Medium: Questions should primarily test the middle two Bloom levels (Applying and Analyzing).\n"
    " - Hard: Questions should primarily test the top two Bloom levels (Evaluating and Creating).\n\n"
    "Your task is to assign a Bloom evaluation score on a scale from 1 to 5 based on how well the question's cognitive "
    "challenge matches the intended difficulty level, where:\n"
    "   (1) Completely misaligned\n"
    "   (2) Mostly misaligned\n"
    "   (3) Moderately aligned\n"
    "   (4) Mostly aligned\n"
    "   (5) Perfectly aligned\n\n"
    "Please follow these steps:\n"
    "1. Read the provided question, knowledge component, and intended difficulty level carefully.\n"
    "2. Recall the six levels of Bloom's Taxonomy and determine which levels the intended difficulty is supposed to target "
    "(Easy targets levels 1–2, Medium targets levels 3–4, and Hard targets levels 5–6).\n"
    "3. Analyze the cognitive demands of the question and compare them with the targeted Bloom levels.\n"
    "4. Assign a Bloom evaluation score (1–5) based on the alignment, and provide a step-by-step explanation of your reasoning.\n\n"
    "Remember to evaluate the question from the perspective of a middle school student, ensuring that the cognitive challenge "
    "is appropriate for that level."
)

# -------------------------------------------------------------------
# Evaluation Functions (each calls GPT with a separate system prompt)
# -------------------------------------------------------------------
def evaluate_qa_clarity(question: str):
    """Evaluate clarity of a question using the CLARITY_SYSTEM_PROMPT."""
    client = OpenAI()

    class EvaluationTemplate(BaseModel):
        reasoning: str
        clarity: int

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": CLARITY_SYSTEM_PROMPT},
            {"role": "user", "content": f"The Question: {question}\n"}
        ],
        response_format=EvaluationTemplate,
    )
    event = completion.choices[0].message.parsed
    event_json = json.loads(event.model_dump_json())
    return event_json

def evaluate_qa_relevance(question: str, kc_name: str, examples: List[str]):
    """Evaluate relevance of a question to a KC and example problems."""
    client = OpenAI()

    class RelevanceEvaluationTemplate(BaseModel):
        reasoning: str
        relevance: int

    user_content = (
        f"The Question: {question}\n"
        f"Knowledge Component: {kc_name}\n"
        f"Examples: {examples}\n"
    )

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": RELEVANCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        response_format=RelevanceEvaluationTemplate,
    )
    event = completion.choices[0].message.parsed
    event_json = json.loads(event.model_dump_json())
    print(event_json)

    return event_json

def evaluate_qa_importance(question: str, kc_name: str):
    """Evaluate importance of a question for a given KC."""
    client = OpenAI()

    class ImportanceEvaluationTemplate(BaseModel):
        reasoning: str
        importance: int

    user_content = (
        f"The Question: {question}\n"
        f"Knowledge Component: {kc_name}\n"
    )

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": IMPORTANCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        response_format=ImportanceEvaluationTemplate,
    )
    event = completion.choices[0].message.parsed
    event_json = json.loads(event.model_dump_json())
    return event_json

def evaluate_qa_difficulty_matching(question: str, kc_name: str, examples: List[str], difficulty: str):
    """Evaluate how well a question matches the specified difficulty level."""
    client = OpenAI()

    class DifficultyMatchingEvaluationTemplate(BaseModel):
        reasoning: str
        difficulty_matching: int

    user_content = (
        f"The Question: {question}\n"
        f"Knowledge Component: {kc_name}\n"
        f"Examples: {examples}\n"
        f"Specified Difficulty Level: {difficulty}\n"
    )

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": DIFFICULTY_MATCHING_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        response_format=DifficultyMatchingEvaluationTemplate,
    )
    event = completion.choices[0].message.parsed
    event_json = json.loads(event.model_dump_json())
    return event_json

def evaluate_qa_answerability(question: str, answer: str, difficulty: str):
    """Evaluate if a question is answerable by a typical middle school student."""
    client = OpenAI()

    class AnswerabilityEvaluationTemplate(BaseModel):
        reasoning: str
        answerability: int

    user_content = (
        f"The Question: {question}\n"
        f"The Answer: {answer}\n"
        f"Specified Difficulty Level: {difficulty}\n"
    )

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": ANSWERABILITY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        response_format=AnswerabilityEvaluationTemplate,
    )
    event = completion.choices[0].message.parsed
    event_json = json.loads(event.model_dump_json())
    return event_json

def evaluate_qa_bloom(question: str, kc_name: str, difficulty: str):
    """Evaluate how well a question aligns with Bloom's Taxonomy for a given difficulty."""
    client = OpenAI()

    class BloomEvaluationTemplate(BaseModel):
        reasoning: str
        bloom_score: int

    user_content = (
        f"The Question: {question}\n"
        f"Knowledge Component: {kc_name}\n"
        f"Intended Difficulty Level: {difficulty}\n"
    )

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": BLOOM_EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        response_format=BloomEvaluationTemplate,
    )
    event = completion.choices[0].message.parsed
    event_json = json.loads(event.model_dump_json())
    return event_json






def evaluate_final_qa(extracted_info, kc: str, all_questions: list, difficulty: str) -> dict:
    """
    Evaluates the final Q&A pair using various evaluation functions.
    
    Args:
      extracted_info: A validated pydantic model instance (ExtractedInfo_Base/Solution/COT).
      kc (str): Knowledge component name.
      all_questions (list): A list of example questions used for relevance/difficulty checks.
      difficulty (str): The difficulty level: "easy", "medium", or "hard".
      
    Returns:
      dict: A dictionary containing the evaluation metrics.
    """
    eval_clarity = evaluate_qa_clarity(extracted_info.question)
    eval_relevance = evaluate_qa_relevance(extracted_info.question, kc, all_questions)
    eval_importance = evaluate_qa_importance(extracted_info.question, kc)
    eval_diff_match = evaluate_qa_difficulty_matching(extracted_info.question, kc, all_questions, difficulty)
    eval_answerable = evaluate_qa_answerability(extracted_info.question, extracted_info.answer, difficulty)
    eval_bloom_score = evaluate_qa_bloom(extracted_info.question, kc, difficulty)

    return {
        "clarity": eval_clarity.get("clarity"),
        "relevance": eval_relevance.get("relevance"),
        "importance": eval_importance.get("importance"),
        "difficulty_matching": eval_diff_match.get("difficulty_matching"),
        "answerability": eval_answerable.get("answerability"),
        "bloom_score": eval_bloom_score.get("bloom_score"),
    }
# -------------------------------------------------------------------
# Function to load data & choose examples by difficulty (optional)
# -------------------------------------------------------------------


def load_question_samples(easy_csv: str, medium_csv: str, hard_csv: str) -> dict:
    """
    Load question samples from easy, medium, and hard CSV files.

    Each CSV file is expected to have a column named "problem_body_clean".
    The function returns a dictionary with:
      - 'easy_questions': 10 random samples from the easy CSV.
      - 'medium_questions': 10 random samples from the medium CSV.
      - 'hard_questions': 10 random samples from the hard CSV.
      - 'all_questions': 10 random samples from the combined data of all three CSV files.

    Args:
      easy_csv (str): File path for the easy questions CSV.
      medium_csv (str): File path for the medium questions CSV.
      hard_csv (str): File path for the hard questions CSV.

    Returns:
      dict: A dictionary with keys 'easy_questions', 'medium_questions', 
            'hard_questions', and 'all_questions'.
    """
    import pandas as pd

    # Load datasets
    df_easy = pd.read_csv(easy_csv)
    df_medium = pd.read_csv(medium_csv)
    df_hard = pd.read_csv(hard_csv)

    # Sample 10 questions from each (using a fixed random_state for reproducibility)
    easy_questions = df_easy["problem_body_clean"].sample(n=10, random_state=42).tolist()
    medium_questions = df_medium["problem_body_clean"].sample(n=10, random_state=42).tolist()
    hard_questions = df_hard["problem_body_clean"].sample(n=10, random_state=42).tolist()

    # Combine all datasets and sample 10 questions from the union
    df_all = pd.concat([df_easy, df_medium, df_hard], ignore_index=True)
    all_questions = df_all["problem_body_clean"].sample(n=10, random_state=42).tolist()

    return {
        "easy_questions": easy_questions,
        "medium_questions": medium_questions,
        "hard_questions": hard_questions,
        "all_questions": all_questions,
    }

# def load_question_datasets(easy_path: str, medium_path: str, hard_path: str):
#     """
#     Utility to load your CSV datasets and return DataFrames
#     for easy, medium, and hard question sets.
#     """
#     df_easy = pd.read_csv(easy_path)
#     df_medium = pd.read_csv(medium_path)
#     df_hard = pd.read_csv(hard_path)

#     return df_easy, df_medium, df_hard

# def get_examples(df_easy, df_medium, df_hard, difficulty: str, num_samples: int = 10) -> List[str]:
#     """
#     Given dataframes for easy, medium, and hard questions,
#     choose 'num_samples' random questions from the specified difficulty
#     and return them as a list of strings.
#     """
#     if difficulty == "easy":
#         df_copy = df_easy["problem_body_clean"].sample(num_samples)
#     elif difficulty == "medium":
#         df_copy = df_medium["problem_body_clean"].sample(num_samples)
#     elif difficulty == "hard":
#         df_copy = df_hard["problem_body_clean"].sample(num_samples)
#     else:
#         raise ValueError("Invalid difficulty level. Must be 'easy', 'medium', or 'hard'.")

#     return list(df_copy.values)

# -------------------------------------------------------------------
# End of workflow_utils.py
# -------------------------------------------------------------------