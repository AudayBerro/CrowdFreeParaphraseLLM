import time
import json

import openai

import lib.prompts_utility 
"""
    The module comprises utility functions designed to facilitate interactions with the OpenAI API.
    These functions streamline various tasks related to the API, enhancing its usability and manageability.
"""

def check_content_structure(response):
    """
    Check if the 'content' item in the JSON object has the desired structure.
    Desired structure: {'paraphrases': ['paraphrase_1', 'paraphrase_2',..., 'paraphrase_n']}

    :args
        response (dict): The JSON object returned by the API call.

    :returns
        bool: True if the 'content' item has the desired structure, False otherwise.

    :Example
        response = {
          "id": "chatcmpl-123",
          "object": "chat.completion",
          "created": 1677652288,
          "choices": [{
            "index": 0,
            "message": {
              "role": "assistant",
              "content": '{"paraphrases": ["paraphrase_1", "paraphrase_2"]}',
            },
            "finish_reason": "stop"
          }],
          "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
          }
        }

        result = check_content_structure(response)
        print(result)  # Output: True
    """
    content = response["choices"][0]["message"]["content"]

    try:
        parsed_content = json.loads(content)
        if isinstance(parsed_content, dict) and "paraphrases" in parsed_content and isinstance(parsed_content["paraphrases"], list):
            return True
    except json.JSONDecodeError:
        pass

    return False

def define_credential(API_KEY):
    """
    The OpenAI API uses API keys for authentication. Visit https://platform.openai.com/account/api-keys page to retrieve the API key you'll use in your requests.
    Remember that your API key is a secret! Do not share it with others or expose it in any client-side code (browsers, apps).
    Production requests must be routed through your own backend server where your API key can be securely loaded from an environment variable or key management service.

    :args
        API_KEY (str): personal API key.
    
    :return
        None
    """
    
    openai.api_key = API_KEY

def generate_paraphrases_with_api(prompt, model="gpt-3.5-turbo", n=1, frequency_penalty=1.5, retry_times=5, retry_delay=5):
    """
    Generate paraphrases using the OpenAI API.

    :args
        prompt (str): The prompt sentence for paraphrase generation.
        model (str): The OpenAI API is powered by a diverse set of models with different capabilities and price points.gpt-3.5-turbo-16k  https://platform.openai.com/docs/models/gpt-3-5
        n (int): How many completions to generate for each prompt.
        frequency_penalty (float): Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far
        retry_times (int): Number of times to retry the API call in case the response doesn't have the desired structure.
        retry_delay (int): Delay in seconds between retry attempts.

    :returns
        List of generated paraphrases.
    """

    messages = [
        {"role": "system", "content": prompt}
    ]

    for _ in range(retry_times + 1):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                frequency_penalty=frequency_penalty,
                n=n,
            )
            
            if check_content_structure(response):
                response_message = json.loads(response["choices"][0]["message"]["content"])
                return response_message["paraphrases"]
            
        except openai.error.Timeout:
            print(f"Request timed out. Retrying in {retry_delay} seconds...")
            pass
        
        except openai.error.ServiceUnavailableError:
            print(f"API server is overloaded or not ready. Retrying in {retry_delay} seconds...")
            pass
        
        time.sleep(retry_delay)

    return []

def generate_paraphrases_with_retry(prompt, max_attempts=10):
    """
    Generate paraphrases using the OpenAI API with retry attempts.

    Parameters:
        prompt (str): The prompt sentence for paraphrase generation.
        max_attempts (int): Maximum number of attempts to retry the API call.

    Returns:
        List of generated paraphrases. If the API call fails after all retry attempts, an empty list is returned.
    """

    attempts = 0
    paraphrases = generate_paraphrases_with_api(prompt)

    while not paraphrases and attempts < max_attempts:
        attempts += 1
        print(f"Attempt {attempts}: The generated list of paraphrases is empty. Retrying...")
        paraphrases = generate_paraphrases_with_api(prompt)

    if not paraphrases:
        print(f"Paraphrase generation failed after {max_attempts} attempts.")
    else:
        print(f"Paraphrases generated successfully after {attempts} attempts.")

    return paraphrases


if __name__ == "__main__":
    seed_utterance = "Book a flight from lyon to Sydney"

    selected_target_pattern = [
        "The weather is nice today.",
        "I love to read books.",
        "The cat is sleeping.",
    ]
    prompt = prompts_utility.get_baseline_prompt(seed_utterance)
    print(prompt)
    a = generate_paraphrases_with_api(prompt)
    print(a)
    print(type(a))