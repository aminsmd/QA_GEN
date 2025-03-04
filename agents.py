from utils import CustomLLMClient_Base, CustomLLMClient_Solution, CustomLLMClient_COT

def generate_system_prompt(agent, autoCOT_flag, difficulty_method, difficulty, kc, 
                           all_questions, easy_questions, medium_questions, hard_questions):
    """
    Generate the system prompt based on agent type and provided parameters.
    
    Parameters:
      agent (str): The type of agent (e.g., "teacher", "generic_critic", "super_critic", "reviser", "student", "ceo").
      autoCOT_flag (bool): Whether to include chain-of-thought instructions.
      difficulty_method (str): One of 'empirical', 'prompting_simple', or 'prompting_empirical'.
      difficulty (str): One of 'easy', 'medium', or 'hard'.
      kc (str): The knowledge component.
      all_questions (str): Questions string for prompting_simple method.
      easy_questions (str): Questions string for easy questions.
      medium_questions (str): Questions string for medium questions.
      hard_questions (str): Questions string for hard questions.
      
    Returns:
      str: The constructed system prompt.
    """
    agent_type = agent.lower()
    
    if agent_type == "teacher":
        # Include chain-of-thought instructions if requested.
        autoCOT_instructions = ""
        if autoCOT_flag:
            autoCOT_instructions = (
                "think step by step and provide your step by step reasoning on how you arrived at your final answer.\n"
            )
        # Base teacher prompt.
        prompt = (
            "You are an experienced middle school educator specializing in mathematics education. "
            f"{autoCOT_instructions}"
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
            'Return your question, solution, and answer without extra commentary.\n'
            '"You can see a few examples below of past questions with various difficulty levels."\n'
        )
        # Append the Questions section based on the difficulty method.
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
        prompt += "\n" + question_section
    elif agent_type == "generic_critic":
        prompt = "You are a Generic Critic. [Placeholder prompt for Generic Critic]."
    elif agent_type == "super_critic":
        prompt = "You are a Super Critic. [Placeholder prompt for Super Critic]."
    elif agent_type == "reviser":
        prompt = "You are a Reviser. [Placeholder prompt for Reviser]."
    elif agent_type in ("student", "student reader"):
        prompt = "You are a Student Reader. [Placeholder prompt for Student Reader]."
    elif agent_type == "ceo":
        prompt = "You are the CEO. [Placeholder prompt for CEO]."
    else:
        prompt = ""
    
    return prompt

def get_agent(agent, seed, temp, autoCOT_flag, Solution_flag, difficulty_method, difficulty, kc, 
              all_questions, easy_questions, medium_questions, hard_questions, config_list, model):
    """
    Returns an agent instance based on the provided parameters.
    
    Parameters:
      agent (str): one of "teacher", "generic_critic", "super_critic", "reviser", "student", "ceo".
      seed: a seed for the LLM configuration.
      temp: temperature for the LLM configuration.
      autoCOT_flag (bool): if True, include chain-of-thought instructions.
      Solution_flag (bool): if True, use the solution client.
      difficulty_method (str): one of 'empirical', 'prompting_simple', or 'prompting_empirical'.
      difficulty (str): one of 'easy', 'medium', or 'hard'.
      kc (str): the knowledge component.
      all_questions (str): questions string for the prompting_simple method.
      easy_questions (str): questions string for easy questions.
      medium_questions (str): questions string for medium questions.
      hard_questions (str): questions string for hard questions.
      config_list (list): configuration list for llm_config.
      model (str): the model to be used.
      
    Returns:
      An instance of AssistantAgent with the proper system prompt and model client registered.
    """
    # Choose the model client based on the flags.
    if Solution_flag:
        model_client_cls = CustomLLMClient_Solution
    elif autoCOT_flag:
        model_client_cls = CustomLLMClient_COT
    else:
        model_client_cls = CustomLLMClient_Solution

    # Generate the system prompt by calling the separate function.
    system_prompt = generate_system_prompt(agent, autoCOT_flag, difficulty_method, difficulty, kc,
                                             all_questions, easy_questions, medium_questions, hard_questions)

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
    agent_instance.register_model_client(model_client_cls=model_client_cls)
    
    return agent_instance
