from utils import CustomLLMClient_Base, CustomLLMClient_Solution, CustomLLMClient_COT, config_list
from autogen import AssistantAgent
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
# (Assuming any other necessary imports, e.g. instructor, are handled elsewhere)

# -------------------------------------------------------------------
# Pydantic Models for parsing teacher responses
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
# Pydantic Models for parsing non-teacher agent responses
# -------------------------------------------------------------------
class ExtractedInfo_GenericCritic(BaseModel):
    feedback: str

class ExtractedInfo_SuperCritic(BaseModel):
    feedback: str

class ExtractedInfo_Reviser(BaseModel):
    feedback: str
    question: str
    answer: str

class ExtractedInfo_Student(BaseModel):
    feedback: str
    answer: str

class ExtractedInfo_CEO(BaseModel):
    question: str
    answer: str

class ExtractedInfo_Versatile(BaseModel):
    feedback: str = None
    question: str = None
    answer: str = None

# -------------------------------------------------------------------
# Function to select the appropriate response model based on agent type and flags
# -------------------------------------------------------------------
def get_extracted_info_model(agent: str, autoCOT_flag: bool, solution_flag: bool):
    agent = agent.lower()
    if agent in ["baseline_teacher", "baseline_teacher_zs", "simple_teacher", "bloom_teacher", "teacher"]:
        if autoCOT_flag:
            return ExtractedInfo_COT
        elif solution_flag:
            return ExtractedInfo_Solution
        else:
            return ExtractedInfo_Base
    elif agent == "generic_critic":
        return ExtractedInfo_GenericCritic
    elif agent == "super_critic":
        return ExtractedInfo_SuperCritic
    elif agent == "reviser":
        return ExtractedInfo_Reviser
    elif agent == "student":
        return ExtractedInfo_Student
    elif agent == "ceo":
        return ExtractedInfo_CEO
    elif agent == "versatile_agent":
        return ExtractedInfo_Versatile
    else:
        return ExtractedInfo_Base

# -------------------------------------------------------------------
# System Prompt Generation
# -------------------------------------------------------------------
def generate_system_prompt(agent, autoCOT_flag, Solution_flag, difficulty_method, difficulty, kc, 
                           all_questions, easy_questions, medium_questions, hard_questions):

    # Compute common variables based on the flags.
    if autoCOT_flag:
        returned_information = "question, answer, & reasoning"
    elif Solution_flag:
        returned_information = "question, answer, solution"
    else:
        returned_information = "question, answer"

    solution_instructions_text = ""
    if Solution_flag:
        solution_instructions_text = "Solve the problem step by step and provide your solution.\n"

    autoCOT_instructions_text = ""
    if autoCOT_flag:
        autoCOT_instructions_text = (
            "Think step by step and provide your step by step reasoning on how you arrived at your final answer.\n"
        )

    # Compute the question section based on difficulty_method.
    if difficulty_method == "empirical":
        question_section = (
            "Questions: \n"
            f"Easy Questions: {easy_questions}\n"
            f"Medium Questions: {medium_questions}\n"
            f"Hard Questions: {hard_questions}"
        )
    elif difficulty_method == "prompting_simple":
        question_section = "Questions: \n" + f"Questions: {all_questions}"
    elif difficulty_method == "prompting_empirical":
        if difficulty.lower() == "easy":
            question_section = "Questions: \n" + f"Questions: {easy_questions}"
        elif difficulty.lower() == "medium":
            question_section = "Questions: \n" + f"Questions: {medium_questions}"
        elif difficulty.lower() == "hard":
            question_section = "Questions: \n" + f"Questions: {hard_questions}"
        else:
            question_section = ""
    else:
        question_section = ""

    # Normalize agent type to lowercase for consistency.
    agent_type = agent.lower()

    # Build the prompt based on the agent type.
    if agent_type == "baseline_teacher":
        prompt = (
            f"You are a middle school math teacher. Your task is to generate a math question in the following Knowledge Component: {kc}\n"
            f"and at the following difficulty level: {difficulty}\n"
            f"Return your {returned_information} without extra commentary.\n"
            f"{solution_instructions_text}"
            f"{autoCOT_instructions_text}"
            "You can see a few examples of questions/problems in the same Knowledge Component:\n"
            f"{question_section}\n"
            "If participating in a group discussion, please review previous messages for context and ensure your response reflects collaborative input."
            "Change the question based on the chat discussion so far.\n"
        )
    elif agent_type == "baseline_teacher_zs":
        prompt = (
            f"You are a middle school math teacher. Your task is to generate a math question in the following Knowledge Component: {kc}\n"
            f"and at the following difficulty level: {difficulty}\n"
            f"Return your {returned_information} without extra commentary.\n"
            f"{solution_instructions_text}"
            f"{autoCOT_instructions_text}\n"
            "If participating in a group discussion, please review previous messages for context and ensure your response reflects collaborative input."
        )
    elif agent_type == "simple_teacher":
        prompt = (
            f"You are a middle school math teacher. Your task is to generate a math question/exercise problem for a student in your course in the following Knowledge Component: {kc}\n"
            f"The question should align with the following difficulty level: {difficulty}\n"
            "Remember to ensure the question is at the correct difficulty for middle school level.\n"
            f"Return your {returned_information} without extra commentary.\n"
            f"{solution_instructions_text}"
            f"{autoCOT_instructions_text}"
            "You can see a few examples of questions/problems in the same Knowledge Component:\n"
            f"{question_section}\n"
            "If participating in a group discussion, please review previous messages for context and ensure your response reflects collaborative input."
        )
    elif agent_type == "bloom_teacher":
        prompt = (
            "You are an experienced middle school educator specializing in mathematics education. "
            f"Your task is to generate one {difficulty} math question on the following knowledge component: {kc}. "
            "The question should align with the intended cognitive challenge level as defined by Bloom's Taxonomy. "
            "Bloom's Taxonomy comprises six levels of learning objectives:\n"
            "1. Remembering\n"
            "2. Understanding\n"
            "3. Applying\n"
            "4. Analyzing\n"
            "5. Evaluating\n"
            "6. Creating\n\n"
            "We define three difficulty levels for generated questions:\n"
            "- Easy: (levels 1-2)\n"
            "- Medium: (levels 3-4)\n"
            "- Hard: (levels 5-6)\n\n"
            "Steps:\n"
            "1. Recall the target Bloom levels.\n"
            "2. Generate one question that aligns with the knowledge component and the targeted Bloom levels.\n"
            "3. Provide a step-by-step solution and the final answer.\n\n"
            "Remember to ensure the question is at the correct difficulty for middle school level.\n"
            f"Return your {returned_information} without extra commentary.\n"
            f"{solution_instructions_text}"
            f"{autoCOT_instructions_text}"
            "You can see a few examples below of past questions with various difficulty levels:\n"
            f"{question_section}\n"
            "If participating in a group discussion, please review previous messages for context and ensure your response reflects collaborative input."
        )
    elif agent_type == "generic_critic":
        prompt = (
            "You are an assistant to a middle school teacher. "
            "Your task is to provide constructive and detailed critical feedback on a math question provided by the teacher.\n"
            "When participating in group discussions, please consider the overall conversation context and previous feedback."
        )
    elif agent_type == "super_critic":
        prompt = (
            "You are an assistant to a middle school teacher. "
            "Your task is to evaluate a math question by offering critical feedback on its clarity, "
            f"its relevance to the Knowledge Component ({kc}), its importance, its alignment with the difficulty level ({difficulty}), "
            "and its overall answerability.\n"
            "If participating in a group discussion, please review previous comments from colleagues before providing your evaluation."
        )
    elif agent_type == "reviser":
        prompt = (
            "You are an assistant to a middle school teacher. "
            "Your task is to review a math question provided by the teacher, revise both the question and its answer, "
            "and offer detailed feedback along with your reasoning for the revisions.\n"
            "In a group discussion context, please incorporate previous discussion and consensus into your revisions."
        )
    elif agent_type == "student":
        prompt = (
            "You are a middle school student assisting your teacher in designing math questions. "
            "Your task is to solve the problem using only middle school-level knowledge, provide feedback on the question from a student's perspective, "
            "and supply the final answer.\n"
            "If participating in a group conversation, please consider the collective feedback before finalizing your response."
        )
    elif agent_type == "ceo":
        prompt = (
            "You are the CEO and the final decision-maker in the process of designing middle school math questions. "
            "You are provided with a set of questions along with example questions below:\n"
            f"{question_section}\n"
            f"Your task is to select the best question based on the required difficulty level ({difficulty}) "
            f"and the relevant Knowledge Component ({kc}).\n"
            "When in a group setting, ensure that your decision reflects the collaborative input from your team."
        )
    elif agent_type == "versatile_agent":
        prompt = (
            "You are a versatile agent participating in group discussions. "
            "Review the questions provided so far, offer constructive feedback, and propose revisions. "
            "If no questions are available, generate a new question based on the required difficulty level "
            f"({difficulty}), the relevant Knowledge Component ({kc}), and the provided examples below:\n"
            f"{question_section}\n"
            "Your main group objective is to reach consensus so your individual goal is to help your team reach a consensus.\n"
            "Remember to review previous messages in the group chat to ensure your feedback aligns with the ongoing discussion."
        )
    else:
        prompt = ""

    return prompt



# -------------------------------------------------------------------
# Base Custom LLM Client (refactored to reduce redundancy)
# -------------------------------------------------------------------
class CustomLLMClient:
    """
    A custom LLM client that uses the instructor library (with an OpenAI backend)
    to parse JSON responses into ExtractedInfo objects.
    """
    response_model = None  # To be defined by subclasses

    def __init__(self, config, **kwargs):
        # Optional: add further configuration or logging here
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
# Specialized LLM Clients for teacher responses
# -------------------------------------------------------------------
class CustomLLMClient_Base(CustomLLMClient):
    response_model = ExtractedInfo_Base

class CustomLLMClient_Solution(CustomLLMClient):
    response_model = ExtractedInfo_Solution

class CustomLLMClient_COT(CustomLLMClient):
    response_model = ExtractedInfo_COT

# -------------------------------------------------------------------
# Specialized LLM Clients for non-teacher agent responses
# -------------------------------------------------------------------
class CustomLLMClient_GenericCritic(CustomLLMClient):
    response_model = ExtractedInfo_GenericCritic

class CustomLLMClient_SuperCritic(CustomLLMClient):
    response_model = ExtractedInfo_SuperCritic

class CustomLLMClient_Reviser(CustomLLMClient):
    response_model = ExtractedInfo_Reviser

class CustomLLMClient_Student(CustomLLMClient):
    response_model = ExtractedInfo_Student

class CustomLLMClient_CEO(CustomLLMClient):
    response_model = ExtractedInfo_CEO

class CustomLLMClient_Versatile(CustomLLMClient):
    response_model = ExtractedInfo_Versatile

# -------------------------------------------------------------------
# Function to return an agent instance based on provided parameters
# -------------------------------------------------------------------
def get_agent(agent, seed, temp, autoCOT_flag, Solution_flag, difficulty_method, difficulty, kc, 
              all_questions, easy_questions, medium_questions, hard_questions, model):
    """
    Returns an agent instance with the proper system prompt and model client registered.
    """
    agent_lower = agent.lower()
    # Select the model client class based on agent type and flags
    if agent_lower in ["teacher", "baseline_teacher", "baseline_teacher_zs", "simple_teacher", "bloom_teacher"]:
        if Solution_flag:
            model_client_cls = CustomLLMClient_Solution
            config_list = [{"model": "custom_llm_extractor", "model_client_cls": "CustomLLMClient_Solution"}]
        elif autoCOT_flag:
            model_client_cls = CustomLLMClient_COT
            config_list = [{"model": "custom_llm_extractor", "model_client_cls": "CustomLLMClient_COT"}]
        else:
            model_client_cls = CustomLLMClient_Base
            config_list = [{"model": "custom_llm_extractor", "model_client_cls": "CustomLLMClient_Base"}]
    elif agent_lower == "generic_critic":
        model_client_cls = CustomLLMClient_GenericCritic
        config_list = [{"model": "custom_llm_extractor", "model_client_cls": "CustomLLMClient_GenericCritic"}]
    elif agent_lower == "super_critic":
        model_client_cls = CustomLLMClient_SuperCritic
        config_list = [{"model": "custom_llm_extractor", "model_client_cls": "CustomLLMClient_SuperCritic"}]
    elif agent_lower == "reviser":
        model_client_cls = CustomLLMClient_Reviser
        config_list = [{"model": "custom_llm_extractor", "model_client_cls": "CustomLLMClient_Reviser"}]
    elif agent_lower in ("student", "student reader"):
        model_client_cls = CustomLLMClient_Student
        config_list = [{"model": "custom_llm_extractor", "model_client_cls": "CustomLLMClient_Student"}]
    elif agent_lower == "ceo":
        model_client_cls = CustomLLMClient_CEO
        config_list = [{"model": "custom_llm_extractor", "model_client_cls": "CustomLLMClient_CEO"}]
    elif agent_lower == "versatile_agent":
        model_client_cls = CustomLLMClient_Versatile
        config_list = [{"model": "custom_llm_extractor", "model_client_cls": "CustomLLMClient_Versatile"}]
    else:
        model_client_cls = CustomLLMClient_Base
        config_list = [{"model": "custom_llm_extractor", "model_client_cls": "CustomLLMClient_Base"}]

    # Generate the system prompt.
    system_prompt = generate_system_prompt(agent, autoCOT_flag,Solution_flag, difficulty_method, difficulty, kc, all_questions, easy_questions, medium_questions, hard_questions)
# -------------------------------------------------------------------
    # system_prompt =  generate_system_prompt(agent, autoCOT_flag, Solution_flag, difficulty_method, difficulty, kc, 
    #                        all_questions, easy_questions, medium_questions, hard_questions)

    # Create the agent instance.
    agent_instance = AssistantAgent(
        name=agent,
        llm_config={
            "config_list": config_list,
            "seed": seed,
            "temperature": temp,
            "cache": None,
            "cache_seed": None,
            "model": model
        },
        system_message=system_prompt
    )
    print(model_client_cls)
    print(config_list)
    agent_instance.register_model_client(model_client_cls=model_client_cls)
    print('after ')
    return agent_instance
