from lib import syntax_tree_utility as syn_utility
from tqdm import tqdm
import numpy as np

"""
    Utility Script

    This is a utility.py script that contains several utility and wrapper functions to process data and avoid code redundancy.
    It provides various helper functions that can be used across different scripts to perform common tasks efficiently.

    Purpose of the Script:
        - Avoid Redundancy: This script aims to encapsulate reusable code blocks, enabling easier and cleaner implementation
        of data processing tasks without duplicating code.
        - Data Processing: The functions in this script are designed to handle data processing tasks, such as parsing,
        filtering, transformation, and other data-related operations.
        - Improving Code Readability: By centralizing utility functions in one script, it promotes code organization and
        enhances the readability of other scripts that utilize these functions.
    
    Author: Auday Berro

    Date: 04/August/2023

    GitHub: https://github.com/AudayBerro
"""
def get_template_for_bertscore(num_paraphrases):
    """
    Get the template column names based on the number of paraphrases to generate.

    :args
        num_paraphrases (int): The number of paraphrases to generate.

    :return
        A list of template column names, e.g., ['p1', 'p2', ...]
    """
    return [f'p{i}' for i in range(1, num_paraphrases + 1)]


def checkgpt_generation_not_empty(paraphrases, length):
    """
    This function help to check if the list of paraphrases returned by ./gpt_utility.generate_paraphrases_with_api() functions is not empty or do not contain any empty paraphrases.

    This function takes a list of strings and checks if it meets certain criteria:
    1. If the list is empty, it will create a new list with NaN values of the desired length.
    2. If the list is not empty but contains empty strings, those empty strings will be replaced with NaN values.
    3. If the list length is less than the desired length, it will be extended with NaN values to match the desired length.

    :args
        paraphrases (list[str]): List of paraphrases generated using the generate_paraphrases_with_api() functions.
        length (int): number of paraphrases in the list of paraphrases.

    :return
        Processed list with no empty strings and length equal to the desired length.
    
    :Example:
        >>> process_string_list(['abc', '', 'def'], 5)
        ['abc', nan, 'def', nan, nan]

        >>> process_string_list([], 3)
        [nan, nan, nan]

        >>> process_string_list(['hello', 'world'], 2)
        ['hello', 'world']
    """
    if not paraphrases:
        return [np.nan] * length

    # Check if any index has an empty string, replace it with NaN
    processed_strings = [p if p else np.nan for p in paraphrases]

    # If the list length is less than the desired length, append NaN values to make it equal
    if len(processed_strings) < length:
        processed_strings.extend([np.nan] * (length - len(processed_strings)))

    return processed_strings

def get_empty_dictionaries(num_keys):
    """
    Generate two empty dictionaries with the specified number of keys.

    :args
        num_keys (int): The number of keys to include in each dictionary.

    :returns
        tuple: A tuple containing two dictionaries with empty lists as values.
    
    :Description
        - p_map (dict): A dictionary with keys "p1", "p2", ..., "pn", where "n" is the number of keys specified by num_keys.
                Each key in p_map corresponds to an empty list that will be used to append generated paraphrases.

        - p_syntax_map (dict): A dictionary with keys "p1", "p2", ..., "pn", where "n" is the number of keys specified by num_keys.
                Each key in p_syntax_map corresponds to an empty list that will be used to append syntax patterns of generated paraphrases.

    :Example
        >>> num_keys = 3
        >>> p_map, p_syntax_map = get_empty_dictionaries(num_keys)
        >>> print(p_map)
        {'p1': [], 'p2': [], 'p3': []}
        >>> print(p_syntax_map)
        {'p1': [], 'p2': [], 'p3': []}
    """
    p_map = {f"p{i+1}": [] for i in range(num_keys)}
    p_syntax_map = {f"p{i+1}": [] for i in range(num_keys)}
    return p_map, p_syntax_map

# ------------- Pattern representation and selection ------------

# Parse Tree Extraction wrapper functions. Help to extract syntax tree
# To capture and control syntax, we follow [Iyyer et al.](https://arxiv.org/pdf/1804.06059.pdf) and define a pattern as the top  two levels of a constituency parse tree.
def extract_syntax_pattern(stanza_model, paraphrase):
    """
    Extract the syntax pattern of the given paraphrase using the Stanza library.
    
    :args
        stanza_model (stanza.pipeline.core.Pipeline): A Stanza Pipeline object for linguistic annotation.
        paraphrase (str): The input sentence to parse.

    :returns
        A tuple containing the syntax pattern, the full syntax tree, and the unrooted syntax tree.
            syntax_pattern (string): defined as the top two levels of the full_syntax_tree
            full_syntax_tree (str): The constituency parse tree of the paraphrase with the 'ROOT' label.
            tree_without_root (str): The constituency parse tree without the 'ROOT' label.

    :raises
        AssertionError: If the root label of the syntax tree is not 'ROOT'

    Example:
    >>> nlp = stanza.Pipeline()
    >>> paraphrase = "She is running."
    >>> syntax_pattern = extract_syntax_pattern(nlp, paraphrase)
    >>> print(syntax_pattern)
    (S (NP) (VP))
    """

    doc = stanza_model(paraphrase)
    full_syntax_tree = doc.sentences[0].constituency
    
    # Remove the ROOT label from the constituency parse tree
    # e.g. ( ROOT ( S ( VP ) ) ) => ( S ( VP ) )
    unrooted_syntax_tree = syn_utility.remove_root_label(str(full_syntax_tree))

    # Get the syntax template of the syntax tree
    syntax_template = syn_utility.get_parse_template(unrooted_syntax_tree, 2)

    return syntax_template,full_syntax_tree,unrooted_syntax_tree

def extract_syntax_patterns_only(stanza_model, df, paraphrase_column):
    """
    Extract only the syntax patterns from the tuples returned by extract_syntax_pattern.

    :args
        stanza_model (Pipeline): The Stanza syntax pattern model instance.
        df (pd.DataFrame): The pandas DataFrame containing the paraphrases.
        paraphrase_column (string): The selected paraphrase column.
    :return
        A list of str of syntax patterns extracted from the tuples.
    """
    syntax_patterns = []
    for paraphrase in tqdm(df[paraphrase_column], position=0, leave=True):#leave=False to ensure that the current progress bar doesn't overwrite the outer loop's progress bar that called this function
        syntax_pattern, _, _ = extract_syntax_pattern(stanza_model, paraphrase)
        syntax_patterns.append(syntax_pattern)
    
    return syntax_patterns

def get_all_syntax_patterns(stanza_model, df, paraphrase_column, new_column_name):
    """
    Get syntax patterns for all paraphrases in the DataFrame using the Stanza syntax pattern model.
    Add to the original DataFrame an additional 'new_column_name' column.

    :args
        stanza_model (Stanza Pipeline): The Stanza syntax pattern model instance.
        df (pd.DataFrame): A DataFrame containing 'id', 'utterance', and 'intent' columns.
        paraphrase_column (string): The selected paraphrase column.
        new_column_name (string): The name of the column that will contains the syntax pattern template.

    :returns
        pd.DataFrame: The original DataFrame with an additional 'paraphrase_template' column.
    """

    # Extract only the syntax patterns from the tuples returned by extract_syntax_pattern
    syntax_patterns = extract_syntax_patterns_only(stanza_model, df, paraphrase_column)

    df[new_column_name] = syntax_patterns

def process_utterance_template(template):
    """
    Process the given utterance template by removing the 'ROOT' label from it.

    This function takes an input syntax pattern template and removes the 'ROOT' label, if present.
    The 'ROOT' label often represents the top-level node in a constituency parse tree, and it is commonly removed
    in natural language processing tasks to focus on the core structure of the utterance.

    :args
        template (str): A syntax pattern template to process.

    :return
        The processed utterance template.
    """
    processed_template = syn_utility.remove_root_label(template)

    return processed_template
# ------------- end Pattern representation and selection utility ------------