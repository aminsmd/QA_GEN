# workflows.py

import time
import os
import pandas as pd

# Import get_agent from agents.py
from agents import get_agent

# Import helper functions and Pydantic models from utils.py
from utils import (
    # Pydantic models:
    ExtractedInfo_Base,
    ExtractedInfo_Solution,
    ExtractedInfo_COT,
    # Helper functions:
    load_question_samples,
    evaluate_final_qa,
    calculate_total_tokens
)



def get_extracted_info_model(autoCOT_flag: bool, solution_flag: bool):
    if autoCOT_flag:
        # If chain-of-thought is requested, we expect the JSON to have a 'reasoning' field.
        return ExtractedInfo_COT
    elif solution_flag:
        # If a solution is requested, we expect the JSON to have 'solution'.
        return ExtractedInfo_Solution
    else:
        # Otherwise, default to question & answer only.
        return ExtractedInfo_Base


def run_SOM1_WF1(
    difficulty: str,
    kc: str,
    n_questions: int = 1,
    num_rounds: int = 2,
    autoCOT: bool = False,
    difficulty_method: str = "empirical",
    easy_csv: str = "/data/text2_easy_questions.csv",
    medium_csv: str = "/data/ext2_medium_questions.csv",
    hard_csv: str = "/data/text2_hard_questions.csv",
    output_csv_path: str = "SOM1_WF1_output.csv",
    seed: int = 42,
    temp: float = 0.7,
    Solution_flag: bool = False,
    model: str = "gpt-4o"
):
    """
    Runs the SOM1/WF1 workflow using:
      - A teacher agent (question_designer)
      - A critic agent (super_critic)
      - A reviser agent

    We load question samples from CSVs, create agents with get_agent() (which uses the given flags),
    run a conversation to refine each question, then evaluate the final results and save to CSV.

    Args:
      difficulty (str): "easy", "medium", or "hard".
      kc (str): Knowledge Component name.
      n_questions (int): Number of distinct questions to generate.
      num_rounds (int): Number of feedback rounds between Critic and Reviser.
      autoCOT (bool): Whether to include chain-of-thought instructions.
      difficulty_method (str): "empirical" or "prompting_simple" or "prompting_empirical".
      output_csv_path (str): Path to save CSV results.
      seed (int): LLM config seed.
      temp (float): LLM config temperature.
      Solution_flag (bool): If True, we expect each response to contain a 'solution' field.
      model (str): The model name or identifier to use in the agent.

    Returns:
      pd.DataFrame: A DataFrame containing the final Q&A and evaluation metrics.
    """
    # 1) Load question samples from CSV
    samples = load_question_samples(easy_csv, medium_csv, hard_csv)
    easy_questions = samples["easy_questions"]
    medium_questions = samples["medium_questions"]
    hard_questions = samples["hard_questions"]
    all_questions = samples["all_questions"]

    # 2) Create the agents using get_agent() from agents.py
    # We'll pass the relevant flags and the chosen model to each agent creation.
    config_list = [{"model": model, "api_key": os.environ.get("OPENAI_API_KEY", "")}]

    question_designer = get_agent(
        agent="teacher",
        seed=seed,
        temp=temp,
        autoCOT_flag=autoCOT,
        Solution_flag=Solution_flag,
        difficulty_method=difficulty_method,
        difficulty=difficulty,
        kc=kc,
        all_questions=all_questions,
        easy_questions=easy_questions,
        medium_questions=medium_questions,
        hard_questions=hard_questions,
        config_list=config_list,
        model=model
    )

    question_reviser = get_agent(
        agent="reviser",
        seed=seed,
        temp=temp,
        autoCOT_flag=autoCOT,
        Solution_flag=Solution_flag,
        difficulty_method=difficulty_method,
        difficulty=difficulty,
        kc=kc,
        all_questions=all_questions,
        easy_questions=easy_questions,
        medium_questions=medium_questions,
        hard_questions=hard_questions,
        config_list=config_list,
        model=model
    )

    critic = get_agent(
        agent="super_critic",
        seed=seed,
        temp=temp,
        autoCOT_flag=autoCOT,
        Solution_flag=Solution_flag,
        difficulty_method=difficulty_method,
        difficulty=difficulty,
        kc=kc,
        all_questions=all_questions,
        easy_questions=easy_questions,
        medium_questions=medium_questions,
        hard_questions=hard_questions,
        config_list=config_list,
        model=model
    )

    # 3) Conversation workflow
    initial_message = {
        "content": "Write one math question for middle school students to solve.",
        "name": "Initializer",
        "role": "user"
    }
    qa_list = []

    # Function to pick the right pydantic model based on flags
    extracted_model_cls = get_extracted_info_model(autoCOT, Solution_flag)

    for i in range(n_questions):
        total_tokens = 0
        messages_chain = []

        # Teacher agent -> initial question
        teacher_response = question_designer.generate_reply(messages=[initial_message])
        # Validate with the appropriate extracted model
        extracted_info = extracted_model_cls.model_validate_json(teacher_response)
        messages_chain.append({
            "content": teacher_response,
            "name": "question_designer",
            "role": "assistant"
        })
        total_tokens += calculate_total_tokens(messages_chain, model=model)

        # num_rounds of critic -> reviser feedback
        for _ in range(num_rounds):
            critic_reply = critic.generate_reply(messages=messages_chain)
            critic_msg = {
                "content": critic_reply,
                "name": "critic",
                "role": "assistant"
            }
            messages_chain.append(critic_msg)
            total_tokens += calculate_total_tokens([critic_msg], model=model)

            reviser_reply = question_reviser.generate_reply(messages=messages_chain)
            extracted_info = extracted_model_cls.model_validate_json(reviser_reply)
            reviser_msg = {
                "content": reviser_reply,
                "name": "question_reviser",
                "role": "assistant"
            }
            messages_chain.append(reviser_msg)
            total_tokens += calculate_total_tokens([reviser_msg], model=model)
            time.sleep(1)  # optional pause

        # 4) Evaluate final Q&A
        eval_metrics = evaluate_final_qa(extracted_info, kc, all_questions, difficulty)
        row_dict = {
            "difficulty": difficulty,
            "kc": kc,
            "question": extracted_info.question,
            "answer": extracted_info.answer,
            **eval_metrics,
            "method": "SOM1/WF1",
            "iteration_tokens": total_tokens,
            "autoCOT": autoCOT,
            "Solution_flag": Solution_flag,
            "difficulty_method": difficulty_method,
        }
        qa_list.append(row_dict)

    # 5) Save & return
    df_results = pd.DataFrame(qa_list)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    return df_results
