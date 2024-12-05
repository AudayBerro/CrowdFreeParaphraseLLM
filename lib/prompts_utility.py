"""
	This script contains a set of convenient helper functions that are specifically designed to build and populate pre-designed prompts with the essential data.
    The script help to populate prompt introduced in:
        Ramirez, Jorge and Jorge and Baez, Marcos and Berro, Auday and Benatallah, Boualem and Casati, Fabio
        "Crowdsourcing Syntactically Diverse Paraphrases with Diversity-Aware Prompts and Workflows."
        International Conference on Advanced Information Systems Engineering. Cham: Springer International Publishing, 2022.
"""

def get_baseline_prompt(utterance, n = 3):
    """
    Generate the prompt for paraphrasing a given utterance.

    :args
        utterance (str): The utterance sentence representing how we will request an API intent in natural language.
        n (int): The number of paraphrases to generate by the prompt.

    :returns
        The baseline prompt, referred to as "condition 1" in Section 4 of the paper.
    """
    paraphrases = {'paraphrases': ['paraphrase_1', 'paraphrase_2',..., f'paraphrase_n']}

    prompt = f"[Instructions]\n"
    prompt += f"Suppose you have an intelligent device such as Amazon Alexa, Apple Siri, or Google Assistant. "
    prompt += f"And this assistant can handle requests like \"find places serving Italian food\".\n"
    prompt += f"Given an original sentence representing a request, we ask you to provide {n} creative and different ways of saying the same request to the intelligent assistant. "
    prompt += f"This task is known as paraphrasing or rewording. We ask you this so that the virtual assistant can understand the many ways a user can express the same request.\n\n"
    prompt += f"The paraphrases should not contain grammatical mistakes, and they should make sense."
    prompt += f"Provide your response in a JSON format. Do not provide any additional information except the JSON."
    prompt += f"Your JSON response should respect this structure:"
    prompt += f"  {paraphrases}\n\n"
    prompt += f"Paraphrase the following original sentence: \"{utterance}\"."

    return prompt

def get_patterns_by_example_prompt(utterance, paraphrases_list, n=3):
    """
    Generate the prompt for paraphrasing a given utterance.

    :args
        utterance (str): The utterance sentence representing how we will request an API intent in natural language.
        paraphrases (List[str]): a list of paraphrases to include in the prompt, guiding GPT to generate paraphrases with matching syntax patterns as those found in the provided paraphrases.
        n (int): The number of paraphrases to generate by the prompt.

    :returns
        The baseline prompt, referred to as "condition 1" in Section 4 of the paper.
    """
    paraphrases = {'paraphrases': ['paraphrase_1', 'paraphrase_2',..., f'paraphrase_n']}

    prompt = f"[Instructions]\n"
    prompt += f"Suppose you have an intelligent device such as Amazon Alexa, Apple Siri, or Google Assistant. "
    prompt += f"And this assistant can handle requests like \"find places serving Italian food\".\n"
    prompt += f"Given an original sentence representing a request, we ask you to provide {n} creative and different ways of saying the same request to the intelligent assistant. "
    prompt += f"This task is known as paraphrasing or rewording. We ask you this so that the virtual assistant can understand the many ways a user can express the same request.\n\n"
    prompt += f"The paraphrases should not contain grammatical mistakes, and they should make sense."
    prompt += f"Provide your response in a JSON format. Do not provide any additional information except the JSON."
    prompt += f"Your JSON response should respect this structure:"
    prompt += f"  {paraphrases}\n\n"
    prompt += f"Paraphrase the following original sentence: \"{utterance}\".\n\n"
    prompt += f"Supply paraphrases inspired by any of the examples:\n"
    prompt += f"N∘\tExample\n"

    for idx, sample in enumerate(paraphrases_list):
        prompt += f"{idx+1}.\t{sample}\n"
    
    prompt += "\n"
    prompt += "• Try to use different words than those in the original sentence.\n"
    prompt += "• Please provide paraphrases whose syntactic structure is different from that of the original sentence but have the same structure as the example shown.\n"
    prompt += "• A different structure may have words from the original sentence moved to another place plus modifications (even adding new words) to make the paraphrase grammatical."
    

    return prompt

def get_taboo_patterns_prompt(utterance, paraphrases_list, n=3):
    """
    Generate the prompt for paraphrasing a given utterance.

    :args
        utterance (str): The utterance sentence representing how we will request an API intent in natural language.
        paraphrases (List[str]): a list of paraphrases to include in the prompt, guiding GPT to generate paraphrases with matching syntax patterns as those found in the provided paraphrases.
        n (int): The number of paraphrases to generate by the prompt.

    :returns
        The baseline prompt, referred to as "condition 1" in Section 4 of the paper.
    """
    paraphrases = {'paraphrases': ['paraphrase_1', 'paraphrase_2',..., f'paraphrase_n']}

    prompt = f"[Instructions]\n"
    prompt += f"Suppose you have an intelligent device such as Amazon Alexa, Apple Siri, or Google Assistant. "
    prompt += f"And this assistant can handle requests like \"find places serving Italian food\".\n"
    prompt += f"Given an original sentence representing a request, we ask you to provide {n} creative and different ways of saying the same request to the intelligent assistant. "
    prompt += f"This task is known as paraphrasing or rewording. We ask you this so that the virtual assistant can understand the many ways a user can express the same request.\n\n"
    prompt += f"The paraphrases should not contain grammatical mistakes, and they should make sense."
    prompt += f"Provide your response in a JSON format. Do not provide any additional information except the JSON."
    prompt += f"Your JSON response should respect this structure:"
    prompt += f"  {paraphrases}\n\n"
    prompt += f"Paraphrase the following original sentence: \"{utterance}\".\n\n"
    prompt += f"Please avoid paraphrases with a structure equal to any of the following examples:\n"
    prompt += f"N∘\tExample\n"

    for idx, sample in enumerate(paraphrases_list):
        prompt += f"{idx+1}.\t{sample}\n"
    
    prompt += "\n"
    prompt += "• Try to use different words than those in the original sentence.\n"
    prompt += "• Please provide paraphrases whose syntactic structure is different from that of the original sentence and the examples.\n"
    

    return prompt