# workflows.py

import time
import os
import pandas as pd

# Import get_agent and get_extracted_info_model from agents.py
from agents import get_agent, get_extracted_info_model

# Import helper functions and Pydantic models from utils.py
from utils import (
    ExtractedInfo_Base,
    ExtractedInfo_Solution,
    ExtractedInfo_COT,
    load_question_samples,
    evaluate_final_qa,
    calculate_total_tokens
)

def run_SOM1_WF1(
    difficulty: str,
    kc: str,
    n_questions: int = 1,
    num_rounds: int = 2,
    autoCOT: bool = False,
    difficulty_method: str = "empirical",
    easy_csv: str = "./data/text2_easy_questions.csv",
    medium_csv: str = "./data/text2_medium_questions.csv",
    hard_csv: str = "./data/text2_hard_questions.csv",
    output_csv_path: str = "./data/SOM1_WF1_output.csv",
    seed: int = 42,
    temp: float = 0.7,
    Solution_flag: bool = False,
    model: str = "gpt-4o"
):
    """
    Runs a three-way chat workflow involving a teacher, a super critic, and a reviser.
    The conversation starts with an initial teacher message and then iterates through rounds:
      critic → reviser → teacher.
    The chat ends with a final teacher reply, which is then extracted for evaluation.
    
    Args:
      difficulty (str): "easy", "medium", or "hard".
      kc (str): Knowledge Component.
      n_questions (int): Number of distinct questions to generate.
      num_rounds (int): Number of rounds (each round includes critic, reviser, then teacher turn).
      autoCOT (bool): Whether to include chain-of-thought instructions.
      difficulty_method (str): "empirical", "prompting_simple", or "prompting_empirical".
      output_csv_path (str): File path to save results.
      seed (int): Seed for LLM configuration.
      temp (float): Temperature for LLM.
      Solution_flag (bool): If True, expect responses with a 'solution' field.
      model (str): Model identifier.
    
    Returns:
      pd.DataFrame: DataFrame containing the final question, answer, and evaluation metrics.
    """
    # 1) Load question samples.
    samples = load_question_samples(easy_csv, medium_csv, hard_csv)
    easy_questions = samples["easy_questions"]
    medium_questions = samples["medium_questions"]
    hard_questions = samples["hard_questions"]
    all_questions = samples["all_questions"]

    # 2) Create agents.
    teacher_agent = get_agent(
        agent="baseline_teacher",
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
        model=model
    )
    critic_agent = get_agent(
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
        model=model
    )
    reviser_agent = get_agent(
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
        model=model
    )

    # Get the appropriate extracted info model for the teacher final output.
    teacher_model_cls = get_extracted_info_model("baseline_teacher", autoCOT, Solution_flag)

    # 3) Conversation workflow.
    initial_message = {
        "content": "Write one math question for middle school students to solve.",
        "name": "Initializer",
        "role": "user"
    }
    qa_list = []

    for i in range(n_questions):
        messages_chain = [initial_message]

        # Initial teacher message.
        teacher_reply = teacher_agent.generate_reply(messages=messages_chain)
        
        teacher_msg = {
            "content": teacher_reply,
            "name": "baseline_teacher",
            "role": "assistant"
        }
        print(teacher_reply)
        messages_chain.append(teacher_msg)

        # Run num_rounds rounds: critic → reviser → teacher.
        for round_idx in range(num_rounds):
            critic_reply = critic_agent.generate_reply(messages=messages_chain)
            critic_msg = {
                "content": critic_reply,
                "name": "super_critic",
                "role": "assistant"
            }
            messages_chain.append(critic_msg)
            print(critic_msg)

            reviser_reply = reviser_agent.generate_reply(messages=messages_chain)
            reviser_msg = {
                "content": reviser_reply,
                "name": "reviser",
                "role": "assistant"
            }
            messages_chain.append(reviser_msg)
            print(reviser_msg)

            teacher_reply = teacher_agent.generate_reply(messages=messages_chain)
            teacher_msg = {
                "content": teacher_reply,
                "name": "baseline_teacher",
                "role": "assistant"
            }
            print(teacher_msg)

            messages_chain.append(teacher_msg)

            time.sleep(1)  # Optional pause between rounds.

        # At the end of the conversation, extract the final teacher output.
        final_teacher_reply = teacher_reply
        final_extracted_info = teacher_model_cls.model_validate_json(final_teacher_reply)

        # Calculate the total token count as the sum of tokens in each message's content.
        total_tokens = calculate_total_tokens(messages_chain, model=model)

        # 4) Evaluate final Q&A.
        eval_metrics = evaluate_final_qa(final_extracted_info, kc, all_questions, difficulty)
        row_dict = {
            "difficulty": difficulty,
            "kc": kc,
            "question": final_extracted_info.question,
            "answer": final_extracted_info.answer,
            **eval_metrics,
            "method": "SOM1/WF1_3Way",
            "iteration_tokens": total_tokens,
            "autoCOT": autoCOT,
            "Solution_flag": Solution_flag,
            "difficulty_method": difficulty_method,
        }
        qa_list.append(row_dict)

    # 5) Save and return results.
    df_results = pd.DataFrame(qa_list)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    return df_results


def run_single_baseline(
    difficulty: str,
    kc: str,
    n_questions: int = 1,
    autoCOT: bool = False,
    difficulty_method: str = "empirical",
    easy_csv: str = "./data/text2_easy_questions.csv",
    medium_csv: str = "./data/text2_medium_questions.csv",
    hard_csv: str = "./data/text2_hard_questions.csv",
    output_csv_path: str = "./data/single_baseline_output.csv",
    seed: int = 42,
    temp: float = 0.7,
    Solution_flag: bool = False,
    model: str = "gpt-4o"
):
    """
    Instantiates a single baseline agent, generates a reply, and evaluates the output.
    
    Args:
      difficulty (str): "easy", "medium", or "hard".
      kc (str): Knowledge Component.
      n_questions (int): Number of questions to generate (currently set up for 1).
      autoCOT (bool): Whether to include chain-of-thought instructions.
      difficulty_method (str): "empirical", "prompting_simple", or "prompting_empirical".
      easy_csv (str): File path for easy questions.
      medium_csv (str): File path for medium questions.
      hard_csv (str): File path for hard questions.
      output_csv_path (str): Where to save the CSV output.
      seed (int): Random seed for consistency.
      temp (float): Temperature parameter for the LLM.
      Solution_flag (bool): If True, expects responses with a 'solution' field.
      model (str): Model identifier.
    
    Returns:
      pd.DataFrame: DataFrame containing the evaluated question, answer, and metrics.
    """
    # 1) Load question samples.
    samples = load_question_samples(easy_csv, medium_csv, hard_csv)
    all_questions = samples["all_questions"]
    easy_questions = samples["easy_questions"]
    medium_questions = samples["medium_questions"]
    hard_questions = samples["hard_questions"]

    # 2) Create the baseline agent.
    baseline_agent = get_agent(
        agent="baseline_teacher",
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
        model=model
    )

    # Get the appropriate extracted info model for the baseline agent.
    teacher_model_cls = get_extracted_info_model("baseline_teacher", autoCOT, Solution_flag)

    # 3) Define the initial message and generate a reply.
    initial_message = {
        "content": "Write one math question for middle school students to solve.",
        "name": "Initializer",
        "role": "user"
    }
    messages_chain = [initial_message]
    
    # Generate the agent's reply.
    baseline_reply = baseline_agent.generate_reply(messages=messages_chain)
    teacher_msg = {
        "content": baseline_reply,
        "name": "baseline_teacher",
        "role": "assistant"
    }
    messages_chain.append(teacher_msg)
    print(teacher_msg)

    # 4) Extract and evaluate the final output.
    final_extracted_info = teacher_model_cls.model_validate_json(baseline_reply)
    total_tokens = calculate_total_tokens(messages_chain, model=model)
    eval_metrics = evaluate_final_qa(final_extracted_info, kc, all_questions, difficulty)
    
    # Prepare the output row.
    row_dict = {
        "difficulty": difficulty,
        "kc": kc,
        "question": final_extracted_info.question,
        "answer": final_extracted_info.answer,
        **eval_metrics,
        "method": "single_baseline",
        "iteration_tokens": total_tokens,
        "autoCOT": autoCOT,
        "Solution_flag": Solution_flag,
        "difficulty_method": difficulty_method,
    }

    # 5) Save and return the result.
    df_result = pd.DataFrame([row_dict])
    df_result.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    return df_result


def run_single_baseline_zs(
    difficulty: str,
    kc: str,
    n_questions: int = 1,
    autoCOT: bool = False,
    difficulty_method: str = "empirical",
    easy_csv: str = "./data/text2_easy_questions.csv",
    medium_csv: str = "./data/text2_medium_questions.csv",
    hard_csv: str = "./data/text2_hard_questions.csv",
    output_csv_path: str = "./data/single_baseline_zs_output.csv",
    seed: int = 42,
    temp: float = 0.7,
    Solution_flag: bool = False,
    model: str = "gpt-4o"
):
    """
    Instantiates a zero-shot baseline teacher agent, generates a reply, and evaluates the output.
    
    Args:
      difficulty (str): "easy", "medium", or "hard".
      kc (str): Knowledge Component.
      n_questions (int): Number of questions to generate (currently set up for 1).
      autoCOT (bool): Whether to include chain-of-thought instructions.
      difficulty_method (str): "empirical", "prompting_simple", or "prompting_empirical".
      easy_csv (str): File path for easy questions.
      medium_csv (str): File path for medium questions.
      hard_csv (str): File path for hard questions.
      output_csv_path (str): Where to save the CSV output.
      seed (int): Random seed for consistency.
      temp (float): Temperature parameter for the LLM.
      Solution_flag (bool): If True, expects responses with a 'solution' field.
      model (str): Model identifier.
    
    Returns:
      pd.DataFrame: DataFrame containing the evaluated question, answer, and metrics.
    """
    # 1) Load question samples.
    samples = load_question_samples(easy_csv, medium_csv, hard_csv)
    all_questions = samples["all_questions"]
    easy_questions = samples["easy_questions"]
    medium_questions = samples["medium_questions"]
    hard_questions = samples["hard_questions"]

    # 2) Create the zero-shot baseline agent.
    baseline_zs_agent = get_agent(
        agent="baseline_teacher_zs",
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
        model=model
    )

    # Get the appropriate extracted info model for the zero-shot baseline agent.
    teacher_model_cls = get_extracted_info_model("baseline_teacher_zs", autoCOT, Solution_flag)

    # 3) Define the initial message and generate a reply.
    initial_message = {
        "content": "Write one math question for middle school students to solve.",
        "name": "Initializer",
        "role": "user"
    }
    messages_chain = [initial_message]
    
    # Generate the agent's reply.
    baseline_zs_reply = baseline_zs_agent.generate_reply(messages=messages_chain)
    teacher_msg = {
        "content": baseline_zs_reply,
        "name": "baseline_teacher_zs",
        "role": "assistant"
    }
    messages_chain.append(teacher_msg)
    print(teacher_msg)

    # 4) Extract and evaluate the final output.
    final_extracted_info = teacher_model_cls.model_validate_json(baseline_zs_reply)
    total_tokens = calculate_total_tokens(messages_chain, model=model)
    eval_metrics = evaluate_final_qa(final_extracted_info, kc, all_questions, difficulty)
    
    # Prepare the output row.
    row_dict = {
        "difficulty": difficulty,
        "kc": kc,
        "question": final_extracted_info.question,
        "answer": final_extracted_info.answer,
        **eval_metrics,
        "method": "single_baseline_zs",
        "iteration_tokens": total_tokens,
        "autoCOT": autoCOT,
        "Solution_flag": Solution_flag,
        "difficulty_method": difficulty_method,
    }

    # 5) Save and return the result.
    df_result = pd.DataFrame([row_dict])
    df_result.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    return df_result


def run_single_bloom_teacher(
    difficulty: str,
    kc: str,
    n_questions: int = 1,
    autoCOT: bool = False,
    difficulty_method: str = "empirical",
    easy_csv: str = "./data/text2_easy_questions.csv",
    medium_csv: str = "./data/text2_medium_questions.csv",
    hard_csv: str = "./data/text2_hard_questions.csv",
    output_csv_path: str = "./data/single_bloom_teacher_output.csv",
    seed: int = 42,
    temp: float = 0.7,
    Solution_flag: bool = False,
    model: str = "gpt-4o"
):
    """
    Instantiates a Bloom teacher agent, generates a reply, and evaluates the output.
    
    Args:
      difficulty (str): "easy", "medium", or "hard".
      kc (str): Knowledge Component.
      n_questions (int): Number of questions to generate (currently set up for 1).
      autoCOT (bool): Whether to include chain-of-thought instructions.
      difficulty_method (str): "empirical", "prompting_simple", or "prompting_empirical".
      easy_csv (str): File path for easy questions.
      medium_csv (str): File path for medium questions.
      hard_csv (str): File path for hard questions.
      output_csv_path (str): Where to save the CSV output.
      seed (int): Random seed for consistency.
      temp (float): Temperature parameter for the LLM.
      Solution_flag (bool): If True, expects responses with a 'solution' field.
      model (str): Model identifier.
    
    Returns:
      pd.DataFrame: DataFrame containing the evaluated question, answer, and metrics.
    """
    # 1) Load question samples.
    samples = load_question_samples(easy_csv, medium_csv, hard_csv)
    all_questions = samples["all_questions"]
    easy_questions = samples["easy_questions"]
    medium_questions = samples["medium_questions"]
    hard_questions = samples["hard_questions"]

    # 2) Create the Bloom teacher agent.
    bloom_teacher_agent = get_agent(
        agent="bloom_teacher",
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
        model=model
    )

    # Get the appropriate extracted info model for the Bloom teacher agent.
    teacher_model_cls = get_extracted_info_model("bloom_teacher", autoCOT, Solution_flag)

    # 3) Define the initial message and generate a reply.
    initial_message = {
        "content": "Write one math question for middle school students to solve.",
        "name": "Initializer",
        "role": "user"
    }
    messages_chain = [initial_message]
    
    # Generate the agent's reply.
    bloom_reply = bloom_teacher_agent.generate_reply(messages=messages_chain)
    teacher_msg = {
        "content": bloom_reply,
        "name": "bloom_teacher",
        "role": "assistant"
    }
    messages_chain.append(teacher_msg)
    print(teacher_msg)

    # 4) Extract and evaluate the final output.
    final_extracted_info = teacher_model_cls.model_validate_json(bloom_reply)
    total_tokens = calculate_total_tokens(messages_chain, model=model)
    eval_metrics = evaluate_final_qa(final_extracted_info, kc, all_questions, difficulty)
    
    # Prepare the output row.
    row_dict = {
        "difficulty": difficulty,
        "kc": kc,
        "question": final_extracted_info.question,
        "answer": final_extracted_info.answer,
        **eval_metrics,
        "method": "single_bloom_teacher",
        "iteration_tokens": total_tokens,
        "autoCOT": autoCOT,
        "Solution_flag": Solution_flag,
        "difficulty_method": difficulty_method,
    }

    # 5) Save and return the result.
    df_result = pd.DataFrame([row_dict])
    df_result.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    return df_result



def run_SOM1(
    difficulty: str,
    kc: str,
    n_questions: int = 1,
    num_rounds: int = 2,
    autoCOT: bool = False,
    difficulty_method: str = "empirical",
    easy_csv: str = "./data/text2_easy_questions.csv",
    medium_csv: str = "./data/text2_medium_questions.csv",
    hard_csv: str = "./data/text2_hard_questions.csv",
    output_csv_path: str = "./data/SOM1_output.csv",
    seed: int = 42,
    temp: float = 0.7,
    Solution_flag: bool = False,
    model: str = "gpt-4o"
):
    """
    Runs a SOM1 workflow where a simple teacher and a super critic exchange messages back and forth.
    
    The conversation is initialized with a user prompt and then proceeds with:
      teacher → critic → teacher ... for a number of rounds.
    
    Args:
      difficulty (str): "easy", "medium", or "hard".
      kc (str): Knowledge Component.
      n_questions (int): Number of distinct questions to generate.
      num_rounds (int): Number of back-and-forth rounds between teacher and critic.
      autoCOT (bool): Whether to include chain-of-thought instructions.
      difficulty_method (str): "empirical", "prompting_simple", or "prompting_empirical".
      easy_csv (str): File path for easy questions.
      medium_csv (str): File path for medium questions.
      hard_csv (str): File path for hard questions.
      output_csv_path (str): File path to save results.
      seed (int): Random seed for consistency.
      temp (float): Temperature for LLM generation.
      Solution_flag (bool): If True, expects responses with a 'solution' field.
      model (str): Model identifier.
    
    Returns:
      pd.DataFrame: DataFrame containing the final question, answer, and evaluation metrics.
    """
    # 1) Load question samples.
    samples = load_question_samples(easy_csv, medium_csv, hard_csv)
    all_questions = samples["all_questions"]
    easy_questions = samples["easy_questions"]
    medium_questions = samples["medium_questions"]
    hard_questions = samples["hard_questions"]

    # 2) Create the agents.
    # Create a simple teacher agent.
    teacher_agent = get_agent(
        agent="simple_teacher",
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
        model=model
    )
    # Create a super critic agent.
    critic_agent = get_agent(
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
        model=model
    )

    # Get the appropriate extracted info model for the teacher output.
    teacher_model_cls = get_extracted_info_model("simple_teacher", autoCOT, Solution_flag)

    # 3) Conversation workflow.
    initial_message = {
        "content": "Write one math question for middle school students to solve.",
        "name": "Initializer",
        "role": "user"
    }
    qa_list = []

    for i in range(n_questions):
        messages_chain = [initial_message]

        # Teacher generates initial reply.
        teacher_reply = teacher_agent.generate_reply(messages=messages_chain)
        teacher_msg = {
            "content": teacher_reply,
            "name": "simple_teacher",
            "role": "assistant"
        }
        messages_chain.append(teacher_msg)
        print("Teacher:", teacher_msg)

        # Conduct the conversation rounds.
        for round_idx in range(num_rounds):
            # Super critic responds.
            critic_reply = critic_agent.generate_reply(messages=messages_chain)
            critic_msg = {
                "content": critic_reply,
                "name": "super_critic",
                "role": "assistant"
            }
            messages_chain.append(critic_msg)
            print("Critic:", critic_msg)

            # Teacher replies after critic feedback.
            teacher_reply = teacher_agent.generate_reply(messages=messages_chain)
            teacher_msg = {
                "content": teacher_reply,
                "name": "simple_teacher",
                "role": "assistant"
            }
            messages_chain.append(teacher_msg)
            print("Teacher:", teacher_msg)

            # Optional pause between rounds.
            time.sleep(1)
        
        # 4) At the end of the conversation, extract and evaluate the final teacher output.
        final_teacher_reply = teacher_reply
        final_extracted_info = teacher_model_cls.model_validate_json(final_teacher_reply)
        total_tokens = calculate_total_tokens(messages_chain, model=model)
        eval_metrics = evaluate_final_qa(final_extracted_info, kc, all_questions, difficulty)

        # Create output row.
        row_dict = {
            "difficulty": difficulty,
            "kc": kc,
            "question": final_extracted_info.question,
            "answer": final_extracted_info.answer,
            **eval_metrics,
            "method": "SOM1",
            "iteration_tokens": total_tokens,
            "autoCOT": autoCOT,
            "Solution_flag": Solution_flag,
            'num_rounds': num_rounds,
            "difficulty_method": difficulty_method,
        }
        qa_list.append(row_dict)

    # 5) Save and return the results.
    df_results = pd.DataFrame(qa_list)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    return df_results


def run_SOM2(
    difficulty: str,
    kc: str,
    n_questions: int = 1,
    num_rounds: int = 2,
    autoCOT: bool = False,
    difficulty_method: str = "empirical",
    easy_csv: str = "./data/text2_easy_questions.csv",
    medium_csv: str = "./data/text2_medium_questions.csv",
    hard_csv: str = "./data/text2_hard_questions.csv",
    output_csv_path: str = "./data/SOM2_output.csv",
    seed: int = 42,
    temp: float = 0.7,
    Solution_flag: bool = False,
    model: str = "gpt-4o"
):
    """
    Runs a SOM2 workflow where a simple teacher and a generic critic exchange messages back and forth.
    
    The conversation starts with an initial prompt, followed by:
      teacher → generic critic → teacher ... for a number of rounds.
    
    Args:
      difficulty (str): "easy", "medium", or "hard".
      kc (str): Knowledge Component.
      n_questions (int): Number of distinct questions to generate.
      num_rounds (int): Number of back-and-forth rounds between teacher and critic.
      autoCOT (bool): Whether to include chain-of-thought instructions.
      difficulty_method (str): "empirical", "prompting_simple", or "prompting_empirical".
      easy_csv (str): File path for easy questions.
      medium_csv (str): File path for medium questions.
      hard_csv (str): File path for hard questions.
      output_csv_path (str): File path to save results.
      seed (int): Random seed for consistency.
      temp (float): Temperature for LLM generation.
      Solution_flag (bool): If True, expects responses with a 'solution' field.
      model (str): Model identifier.
    
    Returns:
      pd.DataFrame: DataFrame containing the final question, answer, and evaluation metrics.
    """
    # 1) Load question samples.
    samples = load_question_samples(easy_csv, medium_csv, hard_csv)
    all_questions = samples["all_questions"]
    easy_questions = samples["easy_questions"]
    medium_questions = samples["medium_questions"]
    hard_questions = samples["hard_questions"]

    # 2) Create the agents.
    # Create a simple teacher agent.
    teacher_agent = get_agent(
        agent="simple_teacher",
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
        model=model
    )
    # Create a generic critic agent.
    critic_agent = get_agent(
        agent="generic_critic",
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
        model=model
    )

    # Get the appropriate extracted info model for the teacher output.
    teacher_model_cls = get_extracted_info_model("simple_teacher", autoCOT, Solution_flag)

    # 3) Conversation workflow.
    initial_message = {
        "content": "Write one math question for middle school students to solve.",
        "name": "Initializer",
        "role": "user"
    }
    qa_list = []

    for i in range(n_questions):
        messages_chain = [initial_message]

        # Teacher generates initial reply.
        teacher_reply = teacher_agent.generate_reply(messages=messages_chain)
        teacher_msg = {
            "content": teacher_reply,
            "name": "simple_teacher",
            "role": "assistant"
        }
        messages_chain.append(teacher_msg)
        print("Teacher:", teacher_msg)

        # Conduct the conversation rounds.
        for round_idx in range(num_rounds):
            # Generic critic responds.
            critic_reply = critic_agent.generate_reply(messages=messages_chain)
            critic_msg = {
                "content": critic_reply,
                "name": "generic_critic",
                "role": "assistant"
            }
            messages_chain.append(critic_msg)
            print("Generic Critic:", critic_msg)

            # Teacher replies after critic feedback.
            teacher_reply = teacher_agent.generate_reply(messages=messages_chain)
            teacher_msg = {
                "content": teacher_reply,
                "name": "simple_teacher",
                "role": "assistant"
            }
            messages_chain.append(teacher_msg)
            print("Teacher:", teacher_msg)

            # Optional pause between rounds.
            time.sleep(1)
        
        # 4) At the end of the conversation, extract and evaluate the final teacher output.
        final_teacher_reply = teacher_reply
        final_extracted_info = teacher_model_cls.model_validate_json(final_teacher_reply)
        total_tokens = calculate_total_tokens(messages_chain, model=model)
        eval_metrics = evaluate_final_qa(final_extracted_info, kc, all_questions, difficulty)

        # Create output row.
        row_dict = {
            "difficulty": difficulty,
            "kc": kc,
            "question": final_extracted_info.question,
            "answer": final_extracted_info.answer,
            **eval_metrics,
            "method": "SOM2",
            "iteration_tokens": total_tokens,
            "autoCOT": autoCOT,
            "Solution_flag": Solution_flag,
            'num_rounds': num_rounds,
            "difficulty_method": difficulty_method,
        }
        qa_list.append(row_dict)

    # 5) Save and return the results.
    df_results = pd.DataFrame(qa_list)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    return df_results


def run_SOM3(
    difficulty: str,
    kc: str,
    n_questions: int = 1,
    num_rounds: int = 2,
    autoCOT: bool = False,
    difficulty_method: str = "empirical",
    easy_csv: str = "./data/text2_easy_questions.csv",
    medium_csv: str = "./data/text2_medium_questions.csv",
    hard_csv: str = "./data/text2_hard_questions.csv",
    output_csv_path: str = "./data/SOM3_output.csv",
    seed: int = 42,
    temp: float = 0.7,
    Solution_flag: bool = False,
    model: str = "gpt-4o"
):
    """
    Runs a SOM3 workflow where three agents—a simple teacher, a reviser, and a generic critic—
    participate in a group chat to refine the teacher's answer.
    
    Conversation flow for each question:
      1. The teacher generates an initial response.
      2. For each round:
           - The generic critic provides feedback.
           - The reviser proposes a revised version.
           - The teacher updates the answer.
    
    Args:
      difficulty (str): "easy", "medium", or "hard".
      kc (str): Knowledge Component.
      n_questions (int): Number of distinct questions to generate.
      num_rounds (int): Number of conversation rounds.
      autoCOT (bool): Whether to include chain-of-thought instructions.
      difficulty_method (str): "empirical", "prompting_simple", or "prompting_empirical".
      easy_csv (str): File path for easy questions.
      medium_csv (str): File path for medium questions.
      hard_csv (str): File path for hard questions.
      output_csv_path (str): Where to save the CSV output.
      seed (int): Random seed for consistency.
      temp (float): Temperature parameter for the LLM.
      Solution_flag (bool): If True, expects responses with a 'solution' field.
      model (str): Model identifier.
    
    Returns:
      pd.DataFrame: DataFrame containing the final question, answer, and evaluation metrics.
    """
    # 1) Load question samples.
    samples = load_question_samples(easy_csv, medium_csv, hard_csv)
    all_questions = samples["all_questions"]
    easy_questions = samples["easy_questions"]
    medium_questions = samples["medium_questions"]
    hard_questions = samples["hard_questions"]

    # 2) Create the agents.
    teacher_agent = get_agent(
        agent="simple_teacher",
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
        model=model
    )
    critic_agent = get_agent(
        agent="generic_critic",
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
        model=model
    )
    reviser_agent = get_agent(
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
        model=model
    )

    # Get the extracted info model for the teacher.
    teacher_model_cls = get_extracted_info_model("simple_teacher", autoCOT, Solution_flag)

    # 3) Conversation workflow.
    initial_message = {
        "content": "Write one math question for middle school students to solve.",
        "name": "Initializer",
        "role": "user"
    }
    qa_list = []

    for i in range(n_questions):
        messages_chain = [initial_message]

        # Teacher generates the initial reply.
        teacher_reply = teacher_agent.generate_reply(messages=messages_chain)
        teacher_msg = {
            "content": teacher_reply,
            "name": "simple_teacher",
            "role": "assistant"
        }
        messages_chain.append(teacher_msg)
        print("Teacher:", teacher_msg)

        # Conduct conversation rounds.
        for round_idx in range(num_rounds):
            # Generic critic provides feedback.
            critic_reply = critic_agent.generate_reply(messages=messages_chain)
            critic_msg = {
                "content": critic_reply,
                "name": "generic_critic",
                "role": "assistant"
            }
            messages_chain.append(critic_msg)
            print("Generic Critic:", critic_msg)

            # Reviser gives a revised version.
            reviser_reply = reviser_agent.generate_reply(messages=messages_chain)
            reviser_msg = {
                "content": reviser_reply,
                "name": "reviser",
                "role": "assistant"
            }
            messages_chain.append(reviser_msg)
            print("Reviser:", reviser_msg)

            # Teacher updates the answer based on the feedback.
            teacher_reply = teacher_agent.generate_reply(messages=messages_chain)
            teacher_msg = {
                "content": teacher_reply,
                "name": "simple_teacher",
                "role": "assistant"
            }
            messages_chain.append(teacher_msg)
            print("Teacher:", teacher_msg)

            time.sleep(1)  # Optional pause between rounds.

        # 4) Extract and evaluate the final teacher output.
        final_teacher_reply = teacher_reply
        final_extracted_info = teacher_model_cls.model_validate_json(final_teacher_reply)
        total_tokens = calculate_total_tokens(messages_chain, model=model)
        eval_metrics = evaluate_final_qa(final_extracted_info, kc, all_questions, difficulty)

        # Prepare the output row.
        row_dict = {
            "difficulty": difficulty,
            "kc": kc,
            "question": final_extracted_info.question,
            "answer": final_extracted_info.answer,
            **eval_metrics,
            "method": "SOM3",
            "iteration_tokens": total_tokens,
            "autoCOT": autoCOT,
            "Solution_flag": Solution_flag,
            'num_rounds': num_rounds,
            "difficulty_method": difficulty_method,
        }
        qa_list.append(row_dict)

    # 5) Save and return the results.
    df_results = pd.DataFrame(qa_list)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    return df_results



def run_SOM4(
    difficulty: str,
    kc: str,
    n_questions: int = 1,
    num_rounds: int = 2,
    autoCOT: bool = False,
    difficulty_method: str = "empirical",
    easy_csv: str = "./data/text2_easy_questions.csv",
    medium_csv: str = "./data/text2_medium_questions.csv",
    hard_csv: str = "./data/text2_hard_questions.csv",
    output_csv_path: str = "./data/SOM4_output.csv",
    seed: int = 42,
    temp: float = 0.7,
    Solution_flag: bool = False,
    model: str = "gpt-4o"
):
    """
    Runs a SOM4 workflow where four agents—baseline_teacher, student, generic_critic, and reviser—
    participate in a group chat to refine the teacher's answer.
    
    Conversation flow for each question:
      1. The baseline_teacher generates an initial response.
      2. For each round:
           - The student provides feedback.
           - The generic critic provides additional feedback.
           - The reviser offers a revised version.
           - The baseline_teacher updates the answer based on the feedback.
           
    Args:
      difficulty (str): "easy", "medium", or "hard".
      kc (str): Knowledge Component.
      n_questions (int): Number of distinct questions to generate.
      num_rounds (int): Number of conversation rounds.
      autoCOT (bool): Whether to include chain-of-thought instructions.
      difficulty_method (str): "empirical", "prompting_simple", or "prompting_empirical".
      easy_csv (str): File path for easy questions.
      medium_csv (str): File path for medium questions.
      hard_csv (str): File path for hard questions.
      output_csv_path (str): File path to save results.
      seed (int): Random seed for consistency.
      temp (float): Temperature for LLM generation.
      Solution_flag (bool): If True, expects responses with a 'solution' field.
      model (str): Model identifier.
    
    Returns:
      pd.DataFrame: DataFrame containing the final question, answer, and evaluation metrics.
    """
    # 1) Load question samples.
    samples = load_question_samples(easy_csv, medium_csv, hard_csv)
    all_questions = samples["all_questions"]
    easy_questions = samples["easy_questions"]
    medium_questions = samples["medium_questions"]
    hard_questions = samples["hard_questions"]

    # 2) Create the agents.
    teacher_agent = get_agent(
        agent="baseline_teacher",
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
        model=model
    )
    student_agent = get_agent(
        agent="student",
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
        model=model
    )
    critic_agent = get_agent(
        agent="generic_critic",
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
        model=model
    )
    reviser_agent = get_agent(
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
        model=model
    )

    # Get the appropriate extracted info model for the teacher.
    teacher_model_cls = get_extracted_info_model("baseline_teacher", autoCOT, Solution_flag)

    # 3) Conversation workflow.
    initial_message = {
        "content": "Write one math question for middle school students to solve.",
        "name": "Initializer",
        "role": "user"
    }
    qa_list = []

    for i in range(n_questions):
        messages_chain = [initial_message]

        # Teacher generates the initial reply.
        teacher_reply = teacher_agent.generate_reply(messages=messages_chain)
        teacher_msg = {
            "content": teacher_reply,
            "name": "baseline_teacher",
            "role": "assistant"
        }
        messages_chain.append(teacher_msg)
        print("Teacher:", teacher_msg)

        # Conduct conversation rounds.
        for round_idx in range(num_rounds):
            # Student provides feedback.
            student_reply = student_agent.generate_reply(messages=messages_chain)
            student_msg = {
                "content": student_reply,
                "name": "student",
                "role": "assistant"
            }
            messages_chain.append(student_msg)
            print("Student:", student_msg)

            # Generic critic provides feedback.
            critic_reply = critic_agent.generate_reply(messages=messages_chain)
            critic_msg = {
                "content": critic_reply,
                "name": "generic_critic",
                "role": "assistant"
            }
            messages_chain.append(critic_msg)
            print("Generic Critic:", critic_msg)

            # Reviser offers a revised version.
            reviser_reply = reviser_agent.generate_reply(messages=messages_chain)
            reviser_msg = {
                "content": reviser_reply,
                "name": "reviser",
                "role": "assistant"
            }
            messages_chain.append(reviser_msg)
            print("Reviser:", reviser_msg)

            # Teacher updates the answer based on the feedback.
            teacher_reply = teacher_agent.generate_reply(messages=messages_chain)
            teacher_msg = {
                "content": teacher_reply,
                "name": "baseline_teacher",
                "role": "assistant"
            }
            messages_chain.append(teacher_msg)
            print("Teacher:", teacher_msg)

            time.sleep(1)  # Optional pause between rounds.

        # 4) Extract and evaluate the final teacher output.
        final_teacher_reply = teacher_reply
        final_extracted_info = teacher_model_cls.model_validate_json(final_teacher_reply)
        total_tokens = calculate_total_tokens(messages_chain, model=model)
        eval_metrics = evaluate_final_qa(final_extracted_info, kc, all_questions, difficulty)

        # Prepare the output row.
        row_dict = {
            "difficulty": difficulty,
            "kc": kc,
            "question": final_extracted_info.question,
            "answer": final_extracted_info.answer,
            **eval_metrics,
            "method": "SOM4",
            "iteration_tokens": total_tokens,
            "autoCOT": autoCOT,
            "Solution_flag": Solution_flag,
            'num_rounds': num_rounds,
            "difficulty_method": difficulty_method,
        }
        qa_list.append(row_dict)

    # 5) Save and return the results.
    df_results = pd.DataFrame(qa_list)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    return df_results


def run_SOM5(
    difficulty: str,
    kc: str,
    n_questions: int = 1,
    num_rounds: int = 2,
    autoCOT: bool = False,
    difficulty_method: str = "empirical",
    easy_csv: str = "./data/text2_easy_questions.csv",
    medium_csv: str = "./data/text2_medium_questions.csv",
    hard_csv: str = "./data/text2_hard_questions.csv",
    output_csv_path: str = "./data/SOM5_output.csv",
    seed: int = 42,
    temp: float = 0.7,
    Solution_flag: bool = False,
    model: str = "gpt-4o"
):
    """
    Runs a SOM5 workflow where five agents—
    baseline_teacher, student, generic_critic, reviser, and ceo—participate
    in a group chat. In each conversation round, the following order is used:
      1. Student provides feedback.
      2. Generic critic offers feedback.
      3. Reviser suggests a revised version.
      4. Baseline teacher updates the answer.
      5. CEO comments on the updated answer.
    The final output is taken from the CEO's message of the last round.

    Args:
      difficulty (str): "easy", "medium", or "hard".
      kc (str): Knowledge Component.
      n_questions (int): Number of distinct questions to generate.
      num_rounds (int): Number of conversation rounds.
      autoCOT (bool): Whether to include chain-of-thought instructions.
      difficulty_method (str): "empirical", "prompting_simple", or "prompting_empirical".
      easy_csv (str): File path for easy questions.
      medium_csv (str): File path for medium questions.
      hard_csv (str): File path for hard questions.
      output_csv_path (str): File path to save results.
      seed (int): Random seed for consistency.
      temp (float): Temperature for LLM generation.
      Solution_flag (bool): If True, expects responses with a 'solution' field.
      model (str): Model identifier.
    
    Returns:
      pd.DataFrame: DataFrame containing the final question, answer, and evaluation metrics.
    """
    # 1) Load question samples.
    samples = load_question_samples(easy_csv, medium_csv, hard_csv)
    all_questions = samples["all_questions"]
    easy_questions = samples["easy_questions"]
    medium_questions = samples["medium_questions"]
    hard_questions = samples["hard_questions"]

    # 2) Create the agents.
    teacher_agent = get_agent(
        agent="baseline_teacher",
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
        model=model
    )
    student_agent = get_agent(
        agent="student",
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
        model=model
    )
    critic_agent = get_agent(
        agent="generic_critic",
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
        model=model
    )
    reviser_agent = get_agent(
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
        model=model
    )
    ceo_agent = get_agent(
        agent="ceo",
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
        model=model
    )

    # Get the extracted info model for the CEO.
    ceo_model_cls = get_extracted_info_model("ceo", autoCOT, Solution_flag)

    # 3) Conversation workflow.
    initial_message = {
        "content": "Write one math question for middle school students to solve.",
        "name": "Initializer",
        "role": "user"
    }
    qa_list = []

    for i in range(n_questions):
        messages_chain = [initial_message]

        # Baseline teacher generates the initial reply.
        teacher_reply = teacher_agent.generate_reply(messages=messages_chain)
        teacher_msg = {
            "content": teacher_reply,
            "name": "baseline_teacher",
            "role": "assistant"
        }
        messages_chain.append(teacher_msg)
        print("Teacher:", teacher_msg)

        # Conversation rounds.
        for round_idx in range(num_rounds):
            # Student provides feedback.
            student_reply = student_agent.generate_reply(messages=messages_chain)
            student_msg = {
                "content": student_reply,
                "name": "student",
                "role": "assistant"
            }
            messages_chain.append(student_msg)
            print("Student:", student_msg)

            # Generic critic provides feedback.
            critic_reply = critic_agent.generate_reply(messages=messages_chain)
            critic_msg = {
                "content": critic_reply,
                "name": "generic_critic",
                "role": "assistant"
            }
            messages_chain.append(critic_msg)
            print("Generic Critic:", critic_msg)

            # Reviser offers a revised version.
            reviser_reply = reviser_agent.generate_reply(messages=messages_chain)
            reviser_msg = {
                "content": reviser_reply,
                "name": "reviser",
                "role": "assistant"
            }
            messages_chain.append(reviser_msg)
            print("Reviser:", reviser_msg)

            # Baseline teacher updates the answer based on the feedback.
            teacher_reply = teacher_agent.generate_reply(messages=messages_chain)
            teacher_msg = {
                "content": teacher_reply,
                "name": "baseline_teacher",
                "role": "assistant"
            }
            messages_chain.append(teacher_msg)
            print("Teacher Update:", teacher_msg)

            # CEO comments on the updated answer.
            ceo_reply = ceo_agent.generate_reply(messages=messages_chain)
            ceo_msg = {
                "content": ceo_reply,
                "name": "ceo",
                "role": "assistant"
            }
            messages_chain.append(ceo_msg)
            print("CEO:", ceo_msg)

            time.sleep(1)  # Optional pause between rounds.

        # 4) Extract and evaluate the final output from the CEO's last message.
        final_extracted_info = ceo_model_cls.model_validate_json(ceo_reply)
        total_tokens = calculate_total_tokens(messages_chain, model=model)
        eval_metrics = evaluate_final_qa(final_extracted_info, kc, all_questions, difficulty)

        # Prepare the output row.
        row_dict = {
            "difficulty": difficulty,
            "kc": kc,
            "question": final_extracted_info.question,
            "answer": final_extracted_info.answer,
            **eval_metrics,
            "method": "SOM5",
            "iteration_tokens": total_tokens,
            "autoCOT": autoCOT,
            "Solution_flag": Solution_flag,
            'num_rounds': num_rounds,
            "difficulty_method": difficulty_method,
        }
        qa_list.append(row_dict)

    # 5) Save and return the results.
    df_results = pd.DataFrame(qa_list)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    return df_results




def run_WF2_parallel(
    difficulty: str,
    kc: str,
    n_questions: int = 1,
    n_critics: int = 3,
    autoCOT: bool = False,
    difficulty_method: str = "empirical",
    easy_csv: str = "./data/text2_easy_questions.csv",
    medium_csv: str = "./data/text2_medium_questions.csv",
    hard_csv: str = "./data/text2_hard_questions.csv",
    output_csv_path: str = "./data/WF2_parallel_output.csv",
    seed: int = 42,
    temp: float = 0.7,
    Solution_flag: bool = False,
    model: str = "gpt-4o"
):
    """
    Runs a WF2_parallel workflow where a baseline teacher generates an initial reply,
    then a specified number of generic critic agents provide parallel feedback.
    All critic feedback is then sent back to the teacher, which produces a final answer
    that is evaluated.

    Conversation flow for each question:
      1. An initial user prompt is provided.
      2. The baseline_teacher generates an initial answer.
      3. n_critics generic_critic agents each independently generate feedback 
         using the teacher’s answer as context.
      4. The teacher then receives all critic feedback and generates a final answer.
      5. The final teacher output is then evaluated.

    Args:
      difficulty (str): "easy", "medium", or "hard".
      kc (str): Knowledge Component.
      n_questions (int): Number of distinct questions to generate.
      n_critics (int): Number of generic critic agents to instantiate.
      autoCOT (bool): Whether to include chain-of-thought instructions.
      difficulty_method (str): "empirical", "prompting_simple", or "prompting_empirical".
      easy_csv (str): File path for easy questions.
      medium_csv (str): File path for medium questions.
      hard_csv (str): File path for hard questions.
      output_csv_path (str): File path to save results.
      seed (int): Random seed for consistency.
      temp (float): Temperature for LLM generation.
      Solution_flag (bool): If True, expects responses with a 'solution' field.
      model (str): Model identifier.

    Returns:
      pd.DataFrame: DataFrame containing the final teacher answer and evaluation metrics.
    """
    # 1) Load question samples.
    samples = load_question_samples(easy_csv, medium_csv, hard_csv)
    all_questions = samples["all_questions"]
    easy_questions = samples["easy_questions"]
    medium_questions = samples["medium_questions"]
    hard_questions = samples["hard_questions"]

    # 2) Create the baseline teacher agent.
    teacher_agent = get_agent(
        agent="simple_teacher",
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
        model=model
    )

    # Create n_critics generic critic agents.
    critic_agents = []
    for i in range(n_critics):
        critic = get_agent(
            agent="generic_critic",
            seed=seed + i,
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
            model=model
        )
        critic_agents.append(critic)

    # Get the extracted info model for the teacher output.
    teacher_model_cls = get_extracted_info_model("baseline_teacher", autoCOT, Solution_flag)

    # 3) Conversation workflow.
    initial_message = {
        "content": "Write one math question for middle school students to solve.",
        "name": "Initializer",
        "role": "user"
    }
    qa_list = []

    for i in range(n_questions):
        messages_chain = [initial_message]

        print(messages_chain)
        # Teacher generates the initial reply.
        initial_teacher_reply = teacher_agent.generate_reply(messages=messages_chain)
        teacher_msg_initial = {
            "content": initial_teacher_reply,
            "name": "baseline_teacher",
            "role": "assistant"
        }
        messages_chain.append(teacher_msg_initial)
        print("Initial Teacher Reply:", teacher_msg_initial)

        # Each critic generates their feedback in parallel.
        for idx, critic_agent in enumerate(critic_agents):
            critic_reply = critic_agent.generate_reply(messages=messages_chain)
            critic_msg = {
                "content": critic_reply,
                "name": f"generic_critic_{idx+1}",
                "role": "assistant"
            }
            messages_chain.append(critic_msg)
            print(f"Generic Critic {idx+1} Reply:", critic_msg)

        # Teacher receives all critic feedback and generates a final answer.
        final_teacher_reply = teacher_agent.generate_reply(messages=messages_chain)
        teacher_msg_final = {
            "content": final_teacher_reply,
            "name": "baseline_teacher",
            "role": "assistant"
        }
        messages_chain.append(teacher_msg_final)
        print("Final Teacher Reply:", teacher_msg_final)

        # 4) Extract and evaluate the final teacher output.
        final_extracted_info = teacher_model_cls.model_validate_json(final_teacher_reply)
        total_tokens = calculate_total_tokens(messages_chain, model=model)
        eval_metrics = evaluate_final_qa(final_extracted_info, kc, all_questions, difficulty)

        # Prepare the output row.
        row_dict = {
            "difficulty": difficulty,
            "kc": kc,
            "question": final_extracted_info.question,
            "answer": final_extracted_info.answer,
            **eval_metrics,
            "method": "WF2_parallel",
            "iteration_tokens": total_tokens,
            "autoCOT": autoCOT,
            "Solution_flag": Solution_flag,
            'n_critics': n_critics,
            "difficulty_method": difficulty_method,
        }
        qa_list.append(row_dict)

    # 5) Save and return the results.
    df_results = pd.DataFrame(qa_list)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    return df_results


def run_WF3_sequential(
    difficulty: str,
    kc: str,
    n_questions: int = 1,
    n_critics: int = 3,
    autoCOT: bool = False,
    difficulty_method: str = "empirical",
    easy_csv: str = "./data/text2_easy_questions.csv",
    medium_csv: str = "./data/text2_medium_questions.csv",
    hard_csv: str = "./data/text2_hard_questions.csv",
    output_csv_path: str = "./data/WF3_sequential_output.csv",
    seed: int = 42,
    temp: float = 0.7,
    Solution_flag: bool = False,
    model: str = "gpt-4o"
):
    """
    Runs a WF3_sequential workflow where a baseline teacher generates an initial answer,
    then a specified number of generic critic agents provide sequential feedback.
    The feedback from one critic is passed on to the next, and finally the teacher
    uses the accumulated conversation to produce a final answer that is evaluated.
    
    Conversation flow for each question:
      1. An initial user prompt is provided.
      2. The baseline_teacher generates an initial answer.
      3. Each of the n_critics generic critics, in sequence, generate feedback based on the conversation so far.
      4. The teacher then receives all the sequential feedback and generates a final answer.
      5. The final teacher output is evaluated.
    
    Args:
      difficulty (str): "easy", "medium", or "hard".
      kc (str): Knowledge Component.
      n_questions (int): Number of distinct questions to generate.
      n_critics (int): Number of generic critic agents to instantiate.
      autoCOT (bool): Whether to include chain-of-thought instructions.
      difficulty_method (str): "empirical", "prompting_simple", or "prompting_empirical".
      easy_csv (str): File path for easy questions.
      medium_csv (str): File path for medium questions.
      hard_csv (str): File path for hard questions.
      output_csv_path (str): File path to save results.
      seed (int): Random seed for consistency.
      temp (float): Temperature for LLM generation.
      Solution_flag (bool): If True, expects responses with a 'solution' field.
      model (str): Model identifier.
    
    Returns:
      pd.DataFrame: DataFrame containing the final teacher answer and evaluation metrics.
    """
    # 1) Load question samples.
    samples = load_question_samples(easy_csv, medium_csv, hard_csv)
    all_questions = samples["all_questions"]
    easy_questions = samples["easy_questions"]
    medium_questions = samples["medium_questions"]
    hard_questions = samples["hard_questions"]

    # 2) Create the baseline teacher agent.
    teacher_agent = get_agent(
        agent="simple_teacher",
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
        model=model
    )
    
    # Create n_critics generic critic agents.
    critic_agents = []
    for i in range(n_critics):
        critic = get_agent(
            agent="generic_critic",
            seed=seed + i,
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
            model=model
        )
        critic_agents.append(critic)
    
    # Get the extracted info model for the teacher output.
    teacher_model_cls = get_extracted_info_model("baseline_teacher", autoCOT, Solution_flag)

    # 3) Conversation workflow.
    initial_message = {
        "content": "Write one math question for middle school students to solve.",
        "name": "Initializer",
        "role": "user"
    }
    qa_list = []
    
    for i in range(n_questions):
        messages_chain = [initial_message]
        
        # Teacher generates the initial reply.
        initial_teacher_reply = teacher_agent.generate_reply(messages=messages_chain)
        teacher_msg_initial = {
            "content": initial_teacher_reply,
            "name": "baseline_teacher",
            "role": "assistant"
        }
        messages_chain.append(teacher_msg_initial)
        print("Initial Teacher Reply:", teacher_msg_initial)
        
        # Sequentially, each critic provides feedback.
        for idx, critic_agent in enumerate(critic_agents):
            critic_reply = critic_agent.generate_reply(messages=messages_chain)
            critic_msg = {
                "content": critic_reply,
                "name": f"generic_critic_{idx+1}",
                "role": "assistant"
            }
            messages_chain.append(critic_msg)
            print(f"Generic Critic {idx+1} Reply:", critic_msg)
        
        # Teacher receives the sequential feedback and generates a final answer.
        final_teacher_reply = teacher_agent.generate_reply(messages=messages_chain)
        teacher_msg_final = {
            "content": final_teacher_reply,
            "name": "baseline_teacher",
            "role": "assistant"
        }
        messages_chain.append(teacher_msg_final)
        print("Final Teacher Reply:", teacher_msg_final)
        
        # 4) Extract and evaluate the final teacher output.
        final_extracted_info = teacher_model_cls.model_validate_json(final_teacher_reply)
        total_tokens = calculate_total_tokens(messages_chain, model=model)
        eval_metrics = evaluate_final_qa(final_extracted_info, kc, all_questions, difficulty)
        
        # Prepare the output row.
        row_dict = {
            "difficulty": difficulty,
            "kc": kc,
            "question": final_extracted_info.question,
            "answer": final_extracted_info.answer,
            **eval_metrics,
            "method": "WF3_sequential",
            "iteration_tokens": total_tokens,
            "autoCOT": autoCOT,
            "Solution_flag": Solution_flag,
            'n_critics': n_critics,
            "difficulty_method": difficulty_method,
        }
        qa_list.append(row_dict)
    
    # 5) Save and return the results.
    df_results = pd.DataFrame(qa_list)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    return df_results


def run_WF4(
    difficulty: str,
    kc: str,
    n_questions: int = 1,
    n_chats: int = 3,
    autoCOT: bool = False,
    difficulty_method: str = "empirical",
    easy_csv: str = "./data/text2_easy_questions.csv",
    medium_csv: str = "./data/text2_medium_questions.csv",
    hard_csv: str = "./data/text2_hard_questions.csv",
    output_csv_path: str = "./data/WF4_output.csv",
    seed: int = 42,
    temp: float = 0.7,
    Solution_flag: bool = False,
    model: str = "gpt-4o"
):
    """
    Runs a WF4 workflow consisting of multiple distinct questions, each processed through multiple independent group chats.
    Each group chat involves three agents: Teacher, Student (as critic), and Generic Critic.
    
    For each question:
      1. Run n_chats independent group chats:
         a. The Teacher generates an initial answer.
         b. The Student provides feedback.
         c. The Generic Critic provides additional feedback.
         d. The Teacher produces a final output for that chat.
      2. Aggregate all teacher outputs from the n_chats into a new conversation.
      3. Pass the aggregated conversation to a CEO agent, which selects the best output.
      4. Evaluate the CEO's final chosen output.
    
    Args:
      difficulty (str): "easy", "medium", or "hard".
      kc (str): Knowledge Component.
      n_questions (int): Number of distinct questions to generate.
      n_chats (int): Number of independent group chats per question.
      autoCOT (bool): Whether to include chain-of-thought instructions.
      difficulty_method (str): "empirical", "prompting_simple", or "prompting_empirical".
      easy_csv (str): File path for easy questions.
      medium_csv (str): File path for medium questions.
      hard_csv (str): File path for hard questions.
      output_csv_path (str): File path to save results.
      seed (int): Random seed for consistency.
      temp (float): Temperature for LLM generation.
      Solution_flag (bool): If True, expects responses with a 'solution' field.
      model (str): Model identifier.
    
    Returns:
      pd.DataFrame: DataFrame containing the CEO's final chosen output for each question and evaluation metrics.
    """
    # 1) Load question samples.
    samples = load_question_samples(easy_csv, medium_csv, hard_csv)
    all_questions = samples["all_questions"]
    easy_questions = samples["easy_questions"]
    medium_questions = samples["medium_questions"]
    hard_questions = samples["hard_questions"]

    # 2) Create agents for the group chats.
    teacher_agent = get_agent(
        agent="baseline_teacher",
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
        model=model
    )
    student_agent = get_agent(
        agent="student",
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
        model=model
    )
    critic_agent = get_agent(
        agent="generic_critic",
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
        model=model
    )
    # CEO agent to aggregate outputs.
    ceo_agent = get_agent(
        agent="ceo",
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
        model=model
    )

    # Get extracted info models.
    teacher_model_cls = get_extracted_info_model("baseline_teacher", autoCOT, Solution_flag)
    ceo_model_cls = get_extracted_info_model("ceo", autoCOT, Solution_flag)

    results = []  # To store output for each question.

    # 3) Outer loop: For each question.
    for q_idx in range(n_questions):
        teacher_outputs = []  # Collect final teacher output from each chat for this question.
        initial_message = {
            "content": "Write one math question for middle school students to solve.",
            "name": "Initializer",
            "role": "user"
        }
        # Run n_chats independent group chats for this question.
        for chat_idx in range(n_chats):
            message_chain = [initial_message]

            # Teacher generates an initial reply.
            teacher_initial = teacher_agent.generate_reply(messages=message_chain)
            teacher_initial_msg = {
                "content": teacher_initial,
                "name": "baseline_teacher",
                "role": "assistant"
            }
            message_chain.append(teacher_initial_msg)
            print(f"Q{q_idx+1} Chat {chat_idx+1} - Teacher Initial:", teacher_initial_msg)

            # Student provides feedback.
            student_feedback = student_agent.generate_reply(messages=message_chain)
            student_msg = {
                "content": student_feedback,
                "name": "student",
                "role": "assistant"
            }
            message_chain.append(student_msg)
            print(f"Q{q_idx+1} Chat {chat_idx+1} - Student Feedback:", student_msg)

            # Generic Critic provides additional feedback.
            critic_feedback = critic_agent.generate_reply(messages=message_chain)
            critic_msg = {
                "content": critic_feedback,
                "name": "generic_critic",
                "role": "assistant"
            }
            message_chain.append(critic_msg)
            print(f"Q{q_idx+1} Chat {chat_idx+1} - Generic Critic Feedback:", critic_msg)

            # Teacher produces a final output for this chat.
            teacher_final = teacher_agent.generate_reply(messages=message_chain)
            teacher_final_msg = {
                "content": teacher_final,
                "name": "baseline_teacher",
                "role": "assistant"
            }
            message_chain.append(teacher_final_msg)
            print(f"Q{q_idx+1} Chat {chat_idx+1} - Teacher Final Output:", teacher_final_msg)

            teacher_outputs.append(teacher_final_msg)

        # 4) Build a new message chain for the CEO for the current question.
        ceo_message_chain = [{
            "content": "Below are teacher outputs from independent group chats. Select the best final output.",
            "name": "Initializer",
            "role": "user"
        }]
        for idx, output in enumerate(teacher_outputs):
            ceo_message_chain.append({
                "content": f"Teacher Output {idx+1}: {output['content']}",
                "name": f"teacher_{idx+1}",
                "role": "assistant"
            })
        print(f"Q{q_idx+1} - CEO Message Chain:", ceo_message_chain)

        # CEO agent selects the best output.
        ceo_reply = ceo_agent.generate_reply(messages=ceo_message_chain)
        ceo_msg = {
            "content": ceo_reply,
            "name": "ceo",
            "role": "assistant"
        }
        ceo_message_chain.append(ceo_msg)
        print(f"Q{q_idx+1} - CEO Final Output:", ceo_msg)

        # 5) Extract and evaluate the final CEO output.
        final_extracted_info = ceo_model_cls.model_validate_json(ceo_reply)
        total_tokens = calculate_total_tokens(ceo_message_chain, model=model)
        eval_metrics = evaluate_final_qa(final_extracted_info, kc, all_questions, difficulty)

        # Prepare the output row for this question.
        row_dict = {
            "difficulty": difficulty,
            "kc": kc,
            "question": final_extracted_info.question,
            "answer": final_extracted_info.answer,
            **eval_metrics,
            "method": "WF4",
            "iteration_tokens": total_tokens,
            "autoCOT": autoCOT,
            "Solution_flag": Solution_flag,
            'n_chats': n_chats,
            "difficulty_method": difficulty_method,
        }
        results.append(row_dict)

    # 6) Save and return the aggregated results.
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    return df_results


def run_WF5(
    difficulty: str,
    kc: str,
    n_questions: int = 1,
    n_chats: int = 3,
    autoCOT: bool = False,
    difficulty_method: str = "empirical",
    easy_csv: str = "./data/text2_easy_questions.csv",
    medium_csv: str = "./data/text2_medium_questions.csv",
    hard_csv: str = "./data/text2_hard_questions.csv",
    output_csv_path: str = "./data/WF5_output.csv",
    seed: int = 42,
    temp: float = 0.7,
    Solution_flag: bool = False,
    model: str = "gpt-4o"
):
    """
    Runs a WF5 workflow consisting of multiple distinct questions, each processed through multiple independent group chats (n_chats).
    Each group chat involves two agents: Teacher and Generic Critic.
    
    For each question:
      1. Run n_chats independent group chats:
         a. The Teacher generates an initial answer.
         b. The Generic Critic provides feedback.
         c. The Teacher updates the answer to produce the final output for that chat.
      2. Aggregate all teacher outputs from the n_chats into a new conversation.
      3. Pass the aggregated conversation to a CEO agent, which selects the best output.
      4. Evaluate the CEO's final chosen output.
    
    Args:
      difficulty (str): "easy", "medium", or "hard".
      kc (str): Knowledge Component.
      n_questions (int): Number of distinct questions to generate.
      n_chats (int): Number of independent group chats per question.
      autoCOT (bool): Whether to include chain-of-thought instructions.
      difficulty_method (str): "empirical", "prompting_simple", or "prompting_empirical".
      easy_csv (str): File path for easy questions.
      medium_csv (str): File path for medium questions.
      hard_csv (str): File path for hard questions.
      output_csv_path (str): File path to save results.
      seed (int): Random seed for consistency.
      temp (float): Temperature for LLM generation.
      Solution_flag (bool): If True, expects responses with a 'solution' field.
      model (str): Model identifier.
    
    Returns:
      pd.DataFrame: DataFrame containing the CEO's final chosen output for each question and evaluation metrics.
    """
    # 1) Load question samples.
    samples = load_question_samples(easy_csv, medium_csv, hard_csv)
    all_questions = samples["all_questions"]
    easy_questions = samples["easy_questions"]
    medium_questions = samples["medium_questions"]
    hard_questions = samples["hard_questions"]

    # 2) Create agents for the group chats.
    teacher_agent = get_agent(
        agent="baseline_teacher",
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
        model=model
    )
    critic_agent = get_agent(
        agent="generic_critic",
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
        model=model
    )
    # CEO agent for aggregating outputs.
    ceo_agent = get_agent(
        agent="ceo",
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
        model=model
    )

    # Get extracted info models.
    teacher_model_cls = get_extracted_info_model("baseline_teacher", autoCOT, Solution_flag)
    ceo_model_cls = get_extracted_info_model("ceo", autoCOT, Solution_flag)

    results = []  # To store the output for each question.

    # 3) Outer loop: For each question.
    for q_idx in range(n_questions):
        teacher_outputs = []  # Collect final teacher output from each chat for this question.
        initial_message = {
            "content": "Write one math question for middle school students to solve.",
            "name": "Initializer",
            "role": "user"
        }
        # For each question, run n_chats independent group chats.
        for chat_idx in range(n_chats):
            message_chain = [initial_message]

            # Teacher generates an initial reply.
            teacher_initial = teacher_agent.generate_reply(messages=message_chain)
            teacher_initial_msg = {
                "content": teacher_initial,
                "name": "baseline_teacher",
                "role": "assistant"
            }
            message_chain.append(teacher_initial_msg)
            print(f"Q{q_idx+1} Chat {chat_idx+1} - Teacher Initial:", teacher_initial_msg)

            # Generic Critic provides feedback.
            critic_feedback = critic_agent.generate_reply(messages=message_chain)
            critic_msg = {
                "content": critic_feedback,
                "name": "generic_critic",
                "role": "assistant"
            }
            message_chain.append(critic_msg)
            print(f"Q{q_idx+1} Chat {chat_idx+1} - Generic Critic Feedback:", critic_msg)

            # Teacher updates the answer based on the critic's feedback.
            teacher_final = teacher_agent.generate_reply(messages=message_chain)
            teacher_final_msg = {
                "content": teacher_final,
                "name": "baseline_teacher",
                "role": "assistant"
            }
            message_chain.append(teacher_final_msg)
            print(f"Q{q_idx+1} Chat {chat_idx+1} - Teacher Final Output:", teacher_final_msg)

            teacher_outputs.append(teacher_final_msg)

        # 4) Build a new message chain for the CEO for the current question.
        ceo_message_chain = [{
            "content": "Below are teacher outputs from independent group chats. Select the best final output.",
            "name": "Initializer",
            "role": "user"
        }]
        for idx, output in enumerate(teacher_outputs):
            ceo_message_chain.append({
                "content": f"Teacher Output {idx+1}: {output['content']}",
                "name": f"teacher_{idx+1}",
                "role": "assistant"
            })
        print(f"Q{q_idx+1} - CEO Message Chain:", ceo_message_chain)

        # CEO agent selects the best output.
        ceo_reply = ceo_agent.generate_reply(messages=ceo_message_chain)
        ceo_msg = {
            "content": ceo_reply,
            "name": "ceo",
            "role": "assistant"
        }
        ceo_message_chain.append(ceo_msg)
        print(f"Q{q_idx+1} - CEO Final Output:", ceo_msg)

        # 5) Extract and evaluate the final CEO output.
        final_extracted_info = ceo_model_cls.model_validate_json(ceo_reply)
        total_tokens = calculate_total_tokens(ceo_message_chain, model=model)
        eval_metrics = evaluate_final_qa(final_extracted_info, kc, all_questions, difficulty)

        # Prepare the output row for this question.
        row_dict = {
            "difficulty": difficulty,
            "kc": kc,
            "question": final_extracted_info.question,
            "answer": final_extracted_info.answer,
            **eval_metrics,
            "method": "WF5",
            "iteration_tokens": total_tokens,
            "autoCOT": autoCOT,
            "Solution_flag": Solution_flag,
            'n_chats': n_chats,
            "difficulty_method": difficulty_method,
        }
        results.append(row_dict)

    # 6) Save and return the aggregated results.
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    return df_results


def run_WF6(
    difficulty: str,
    kc: str,
    n_questions: int = 1,
    n_chats: int = 3,
    autoCOT: bool = False,
    difficulty_method: str = "empirical",
    easy_csv: str = "./data/text2_easy_questions.csv",
    medium_csv: str = "./data/text2_medium_questions.csv",
    hard_csv: str = "./data/text2_hard_questions.csv",
    output_csv_path: str = "./data/WF6_output.csv",
    seed: int = 42,
    temp: float = 0.7,
    Solution_flag: bool = False,
    model: str = "gpt-4o"
):
    """
    Runs a WF6 workflow consisting of multiple distinct questions, each processed through multiple independent group chats (n_chats).
    Each group chat involves four agents: baseline_teacher, student (as critic), generic_critic, and reviser.
    
    For each question:
      1. Run n_chats independent group chats:
         a. The teacher generates an initial answer.
         b. The student provides feedback.
         c. The generic critic adds further feedback.
         d. The reviser offers a revised version.
         e. The teacher updates the answer to produce the final output for that chat.
      2. Aggregate all teacher outputs from the n_chats into a new conversation.
      3. Pass the aggregated conversation to a CEO agent, which selects the best output.
      4. Evaluate the CEO's final chosen output.
    
    Args:
      difficulty (str): "easy", "medium", or "hard".
      kc (str): Knowledge Component.
      n_questions (int): Number of distinct questions to generate.
      n_chats (int): Number of independent group chats per question.
      autoCOT (bool): Whether to include chain-of-thought instructions.
      difficulty_method (str): "empirical", "prompting_simple", or "prompting_empirical".
      easy_csv (str): File path for easy questions.
      medium_csv (str): File path for medium questions.
      hard_csv (str): File path for hard questions.
      output_csv_path (str): File path to save results.
      seed (int): Random seed for consistency.
      temp (float): Temperature for LLM generation.
      Solution_flag (bool): If True, expects responses with a 'solution' field.
      model (str): Model identifier.
    
    Returns:
      pd.DataFrame: DataFrame containing the CEO's final chosen output for each question and evaluation metrics.
    """
    # 1) Load question samples.
    samples = load_question_samples(easy_csv, medium_csv, hard_csv)
    all_questions = samples["all_questions"]
    easy_questions = samples["easy_questions"]
    medium_questions = samples["medium_questions"]
    hard_questions = samples["hard_questions"]

    # 2) Create agents for group chats.
    teacher_agent = get_agent(
        agent="baseline_teacher",
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
        model=model
    )
    student_agent = get_agent(
        agent="student",
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
        model=model
    )
    critic_agent = get_agent(
        agent="generic_critic",
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
        model=model
    )
    reviser_agent = get_agent(
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
        model=model
    )
    # CEO agent for aggregating outputs.
    ceo_agent = get_agent(
        agent="ceo",
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
        model=model
    )

    # Get extracted info models.
    teacher_model_cls = get_extracted_info_model("baseline_teacher", autoCOT, Solution_flag)
    ceo_model_cls = get_extracted_info_model("ceo", autoCOT, Solution_flag)

    results = []  # to store output for each question

    # 3) Outer loop: for each question.
    for q_idx in range(n_questions):
        teacher_outputs = []  # collect final teacher output from each chat for this question
        initial_message = {
            "content": "Write one math question for middle school students to solve.",
            "name": "Initializer",
            "role": "user"
        }
        # For each question, run n_chats independent group chats.
        for chat_idx in range(n_chats):
            message_chain = [initial_message]

            # Teacher generates an initial reply.
            teacher_initial = teacher_agent.generate_reply(messages=message_chain)
            teacher_initial_msg = {
                "content": teacher_initial,
                "name": "baseline_teacher",
                "role": "assistant"
            }
            message_chain.append(teacher_initial_msg)
            print(f"Q{q_idx+1} Chat {chat_idx+1} - Teacher Initial:", teacher_initial_msg)

            # Student provides feedback.
            student_feedback = student_agent.generate_reply(messages=message_chain)
            student_msg = {
                "content": student_feedback,
                "name": "student",
                "role": "assistant"
            }
            message_chain.append(student_msg)
            print(f"Q{q_idx+1} Chat {chat_idx+1} - Student Feedback:", student_msg)

            # Generic Critic provides additional feedback.
            critic_feedback = critic_agent.generate_reply(messages=message_chain)
            critic_msg = {
                "content": critic_feedback,
                "name": "generic_critic",
                "role": "assistant"
            }
            message_chain.append(critic_msg)
            print(f"Q{q_idx+1} Chat {chat_idx+1} - Generic Critic Feedback:", critic_msg)

            # Reviser offers a revised version.
            reviser_feedback = reviser_agent.generate_reply(messages=message_chain)
            reviser_msg = {
                "content": reviser_feedback,
                "name": "reviser",
                "role": "assistant"
            }
            message_chain.append(reviser_msg)
            print(f"Q{q_idx+1} Chat {chat_idx+1} - Reviser Feedback:", reviser_msg)

            # Teacher updates the answer based on all feedback.
            teacher_final = teacher_agent.generate_reply(messages=message_chain)
            teacher_final_msg = {
                "content": teacher_final,
                "name": "baseline_teacher",
                "role": "assistant"
            }
            message_chain.append(teacher_final_msg)
            print(f"Q{q_idx+1} Chat {chat_idx+1} - Teacher Final Output:", teacher_final_msg)

            teacher_outputs.append(teacher_final_msg)

        # 4) Build a new message chain for the CEO for the current question.
        ceo_message_chain = [{
            "content": "Below are teacher outputs from independent group chats. Select the best final output.",
            "name": "Initializer",
            "role": "user"
        }]
        for idx, output in enumerate(teacher_outputs):
            ceo_message_chain.append({
                "content": f"Teacher Output {idx+1}: {output['content']}",
                "name": f"teacher_{idx+1}",
                "role": "assistant"
            })
        print(f"Q{q_idx+1} - CEO Message Chain:", ceo_message_chain)

        # CEO agent selects the best output.
        ceo_reply = ceo_agent.generate_reply(messages=ceo_message_chain)
        ceo_msg = {
            "content": ceo_reply,
            "name": "ceo",
            "role": "assistant"
        }
        ceo_message_chain.append(ceo_msg)
        print(f"Q{q_idx+1} - CEO Final Output:", ceo_msg)

        # 5) Extract and evaluate the final CEO output.
        final_extracted_info = ceo_model_cls.model_validate_json(ceo_reply)
        total_tokens = calculate_total_tokens(ceo_message_chain, model=model)
        eval_metrics = evaluate_final_qa(final_extracted_info, kc, all_questions, difficulty)

        # Prepare the output row for this question.
        row_dict = {
            "difficulty": difficulty,
            "kc": kc,
            "question": final_extracted_info.question,
            "answer": final_extracted_info.answer,
            **eval_metrics,
            "method": "WF6",
            "iteration_tokens": total_tokens,
            "autoCOT": autoCOT,
            "Solution_flag": Solution_flag,
            'n_chats': n_chats,
            "difficulty_method": difficulty_method,
        }
        results.append(row_dict)

    # 6) Save and return the aggregated results.
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    return df_results


import random

def run_society_of_minds(
    difficulty: str,
    kc: str,
    n_questions: int = 1,
    n_agents: int = 3,
    num_rounds: int = 2,
    autoCOT: bool = False,
    difficulty_method: str = "empirical",
    easy_csv: str = "./data/text2_easy_questions.csv",
    medium_csv: str = "./data/text2_medium_questions.csv",
    hard_csv: str = "./data/text2_hard_questions.csv",
    output_csv_path: str = "./data/society_of_minds_output.csv",
    seed: int = 42,
    Solution_flag: bool = False,
    model: str = "gpt-4o"
):
    """
    Runs a Society of Minds workflow where:
      - A simple_teacher initiates the conversation.
      - n_agents "versatile" agents (each instantiated with a randomly chosen seed and temperature)
        participate in the chat by taking turns over num_rounds rounds.
      - The complete conversation is then passed to a CEO agent, which provides the final output.
    
    Args:
      difficulty (str): "easy", "medium", or "hard".
      kc (str): Knowledge Component.
      n_questions (int): Number of distinct questions to generate.
      n_agents (int): Number of versatile agents to instantiate.
      num_rounds (int): Number of rounds of conversation.
      autoCOT (bool): Whether to include chain-of-thought instructions.
      difficulty_method (str): "empirical", "prompting_simple", or "prompting_empirical".
      easy_csv (str): Path to the CSV file for easy questions.
      medium_csv (str): Path to the CSV file for medium questions.
      hard_csv (str): Path to the CSV file for hard questions.
      output_csv_path (str): Path to save the output CSV.
      seed (int): Seed for randomization.
      Solution_flag (bool): If True, expects responses with a 'solution' field.
      model (str): Model identifier.
    
    Returns:
      pd.DataFrame: DataFrame containing the CEO's final output and evaluation metrics for each question.
    """
    # Set seed for reproducibility.
    random.seed(seed)
    
    # 1) Load question samples.
    samples = load_question_samples(easy_csv, medium_csv, hard_csv)
    all_questions = samples["all_questions"]
    easy_questions = samples["easy_questions"]
    medium_questions = samples["medium_questions"]
    hard_questions = samples["hard_questions"]

    # 2) Create the simple_teacher agent.
    teacher_agent = get_agent(
        agent="simple_teacher",
        seed=seed,
        temp=0.7,  # Fixed temperature for the teacher.
        autoCOT_flag=autoCOT,
        Solution_flag=Solution_flag,
        difficulty_method=difficulty_method,
        difficulty=difficulty,
        kc=kc,
        all_questions=all_questions,
        easy_questions=easy_questions,
        medium_questions=medium_questions,
        hard_questions=hard_questions,
        model=model
    )

    # 3) Create n_agents versatile agents with random seeds and random temperatures.
    versatile_agents = []
    for i in range(n_agents):
        random_seed = random.randint(1, 1000000)
        random_temp = random.uniform(0.01, 0.99)  # Random temperature between 0.7 and 1.0.
        agent = get_agent(
            agent="versatile_agent",
            seed=random_seed,
            temp=random_temp,
            autoCOT_flag=autoCOT,
            Solution_flag=Solution_flag,
            difficulty_method=difficulty_method,
            difficulty=difficulty,
            kc=kc,
            all_questions=all_questions,
            easy_questions=easy_questions,
            medium_questions=medium_questions,
            hard_questions=hard_questions,
            model=model
        )
        versatile_agents.append(agent)

    # 4) Create the CEO agent.
    ceo_agent = get_agent(
        agent="ceo",
        seed=seed,
        temp=0.7,
        autoCOT_flag=autoCOT,
        Solution_flag=Solution_flag,
        difficulty_method=difficulty_method,
        difficulty=difficulty,
        kc=kc,
        all_questions=all_questions,
        easy_questions=easy_questions,
        medium_questions=medium_questions,
        hard_questions=hard_questions,
        model=model
    )
    
    # Get the extracted info model for the CEO output.
    ceo_model_cls = get_extracted_info_model("ceo", autoCOT, Solution_flag)

    results = []  # To store outputs for each question.
    initial_message = {
        "content": "Write one math question for middle school students to solve.",
        "name": "Initializer",
        "role": "user"
    }

    # 5) Process each question.
    for q_idx in range(n_questions):
        message_chain = [initial_message]

        # The simple_teacher starts the conversation.
        teacher_reply = teacher_agent.generate_reply(messages=message_chain)
        teacher_msg = {
            "content": teacher_reply,
            "name": "simple_teacher",
            "role": "assistant"
        }
        message_chain.append(teacher_msg)
        print(f"Question {q_idx+1} - Teacher:", teacher_msg)

        # For the specified number of rounds, each versatile agent takes a turn.
        for round_idx in range(num_rounds):
            for idx, agent in enumerate(versatile_agents):
                agent_reply = agent.generate_reply(messages=message_chain)
                agent_msg = {
                    "content": agent_reply,
                    "name": f"versatile_{idx+1}",
                    "role": "assistant"
                }
                message_chain.append(agent_msg)
                print(f"Q{q_idx+1} Round {round_idx+1} - Versatile Agent {idx+1}:", agent_msg)
                # Optionally, you might want to pause here (e.g., time.sleep(1)).

        # 6) Pass the complete conversation to the CEO agent.
        ceo_reply = ceo_agent.generate_reply(messages=message_chain)
        ceo_msg = {
            "content": ceo_reply,
            "name": "ceo",
            "role": "assistant"
        }
        message_chain.append(ceo_msg)
        print(f"Question {q_idx+1} - CEO Final Output:", ceo_msg)

        # 7) Extract and evaluate the final output.
        final_extracted_info = ceo_model_cls.model_validate_json(ceo_reply)
        total_tokens = calculate_total_tokens(message_chain, model=model)
        eval_metrics = evaluate_final_qa(final_extracted_info, kc, all_questions, difficulty)

        row_dict = {
            "difficulty": difficulty,
            "kc": kc,
            "question": final_extracted_info.question,
            "answer": final_extracted_info.answer,
            **eval_metrics,
            "method": "society_of_minds",
            "iteration_tokens": total_tokens,
            "autoCOT": autoCOT,
            "Solution_flag": Solution_flag,
            "n_agents": n_agents,
            "num_rounds": num_rounds,
            "difficulty_method": difficulty_method,
        }
        results.append(row_dict)

    # 8) Save and return the results.
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    return df_results
