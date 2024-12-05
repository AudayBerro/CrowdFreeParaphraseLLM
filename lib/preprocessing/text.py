import os

from typing import List
import stanza
import numpy as np
import string

from nltk import tokenize, ParentedTree
from nltk.stem import WordNetLemmatizer

"""
I modified this script to be able to run it on my pc, the skeleton of this code was written by Jorge, he extarct parse tree using coreNLP server
Instead i modified the code to etarct syntax treee using stanza library.
Handy helper functions for dealing with text.

based on code from API2CAN and automatedParaphrase
"""

__all__ = ["wordnet_lemmatize", "corenlp_lemmatize", "get_constituency_parse_tree","get_parse_template","get_full_parse","remove_root_label"]

def remove_root_label(tree_str):
    """
    Removes the ROOT label and its corresponding pair of brackets from the given syntax tree string.

    :args
        tree_str (str): The input syntax tree string to be cleaned.

    :returns
        str: The cleaned syntax tree string with the ROOT label and its brackets removed.

    :example
        >>> tree_str = "( ROOT ( FRAG ( NP ) ( ADVP ) ) )"
        >>> cleaned_tree_str = remove_root_label(tree_str)
        >>> print(cleaned_tree_str)
        ( FRAG ( NP ) ( ADVP ) )

    :Note
        This function assumes that the input syntax tree string is well-formed and contains a ROOT label.
        If the input is not as expected, the behavior is undefined.
    """

    # Check if the input tree_str contains a well-formed ROOT label
    if "ROOT" not in tree_str:
        return tree_str

    # Find the index of the first '(' and last ')'
    first_open_bracket_idx = tree_str.find('(')
    last_close_bracket_idx = tree_str.rfind(')')

    # Remove the first '(' and the last ')' and their corresponding pairs
    cleaned_tree_str = tree_str[first_open_bracket_idx+1:last_close_bracket_idx]

    # Find the index of the first space to remove the ROOT label
    first_space_idx = cleaned_tree_str.find(' ')
    cleaned_tree_str = cleaned_tree_str[first_space_idx+1:]

    return cleaned_tree_str

def get_constituency_parse_tree(nlp,utterance):
    """
    Get the constituency parse tree of the given utterance using the Stanza library.
    Source: /home/MyPC/Desktop/vitorCollaboration/process_amadeus.py

    :args
        nlp (stanza.pipeline.core.Pipeline): A Stanza Pipeline object for linguistic annotation.
        utterance (str): The input utterance to parse.

    :returns
        tree (stanza.models.constituency.parse_tree.Tree): The constituency parse tree of the utterance.
        root (str): The root label of the parse tree (should be 'ROOT').
        tree_without_root (Tree): The parse tree without the root label.

    :raises
        AssertionError: If the root label of the tree is not 'ROOT'.
    """

    doc = nlp(utterance)
    #print(f"doc.sentences: {doc.sentences[0].constituency}")
    tree = doc.sentences[0].constituency

    return str(tree)

def wordnet_lemmatize(sentence):
    lemmatizer = WordNetLemmatizer()
    ret = []
    for t in tokenize.word_tokenize(sentence):
        ret.append(lemmatizer.lemmatize(t))
    return " ".join(ret)


def corenlp_lemmatize(nlp,sentence: str) -> str:
    doc = nlp(sentence)
    # lemmas = [w.lemma for w in doc.sentences[0].words]
    lemmas = []

    for s in doc.sentences:
        for w in s.words:
            lemmas.append(w.lemma)
    lem_str = ""

    for w in lemmas:
        if not _is_punctuation(w):
            lem_str += " "
        lem_str += w
    return lem_str.strip()


def get_parse_template(sentence: str, lemmatize=False) -> List[str]:
    """
    Obtain a parse template as defined in "Adversarial Example Generation with Syntactically Controlled Paraphrase Networks".
    Basically this function returns the top two levels of the constituency parse tree.
    """
    if lemmatize:
        sentence = corenlp_lemmatize(sentence)
    #parsed = corenlp_parse(sentence)
    parsed = get_constituency_parse_tree(nlp,sentence)
    parsed_tree = ParentedTree.fromstring(parsed)
    _parse_tree_level_dropout(parsed_tree)
    template = _deleaf(parsed_tree)
    return template

def get_full_parse(nlp,sentence: str) -> List[str]:    
    parsed = get_constituency_parse_tree(nlp,sentence)
    parsed_tree = ParentedTree.fromstring(parsed)
    template = _deleaf(parsed_tree)
    return template

# ------------- internal ------------
def _deleaf(tree: ParentedTree) -> List[str]:
    """
    Removes leaf nodes (i.e. words) from the given tree.

    Adapted from https://github.com/miyyer/scpn/blob/master/scpn_utils.py
    """
    nonleaves = ""
    for w in str(tree).replace("\n", "").split():
        w = w.replace("(", "( ").replace(")", " )")
        nonleaves += w + " "

    arr = nonleaves.split()
    for n, i in enumerate(arr):
        if n + 1 < len(arr):
            tok1 = arr[n]
            tok2 = arr[n + 1]
            if not _is_paren(tok1) and not _is_paren(tok2):
                arr[n + 1] = ""

    nonleaves = " ".join(arr)
    return nonleaves.split()


def _is_paren(tok):
    return tok == ")" or tok == "("


def _parse_tree_level_dropout(tree: ParentedTree, level=2) -> None:
    """
    Removes (in-place) levels of parse tree below the specificied level.

    Adapted from https://github.com/miyyer/scpn/blob/master/scpn_utils.py
    """

    def parse_tree_level_dropout2(tree, level, mlevel):
        if level == mlevel:
            for idx, n in enumerate(tree):
                if isinstance(n, ParentedTree):
                    tree[idx] = "(" + n.label() + ")"
        else:
            for n in tree:
                parse_tree_level_dropout2(n, level + 1, mlevel)

    parse_tree_level_dropout2(tree, 1, level)


def _is_punctuation(txt: str) -> bool:
    return txt in string.punctuation
