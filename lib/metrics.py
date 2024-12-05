"""
Metrics relevant to paraphrasing

code from diversity_metrics.py @ automatedParaphraase
"""
# to make it work on Heroku, we use nltk.txt file as specified here https://devcenter.heroku.com/articles/python-nltk
import re
import nltk
from typing import Any, Dict, List
from nltk import tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu

from lib.preprocessing.text import wordnet_lemmatize


def compute_ttr(input_utterance: str, paraphrases: List[str]) -> Dict[str, Any]:
    """
    Type-Token-Ratio calculates the ratio of unique words to the total number of words in the utterances
    """
    input_tokens = tokenize.word_tokenize(__pre_process(input_utterance))
    vset = set(input_tokens)
    total_words = len(input_tokens)
    total_unique_words = 0

    for p in paraphrases:
        p = __pre_process(p)
        tokens = tokenize.word_tokenize(p)
        total_words += len(tokens)
        vset = vset.union(set(tokens))
    total_unique_words += len(vset)
    ttr_score = len(vset) / total_words
    return {
        "total_words": total_words,
        "total_unique_words": total_unique_words,
        "ttr": ttr_score,
    }

# TODO: change this to the corpus-pinc
def compute_pinc(input_utterance: str, paraphrase: str, n=4) -> float:
    """
    Paraphrase In N-gram Changes measures the percentage of n-gram changes between the initial utterance (a) and a collected utterance (b)
    :param input_utterance: sentence a
    :param paraphrase: sentence b
    :param n: n-grams, default n=4 e.g a=[1,2,3] b=[2,3,4] => 2-grams of a: {(1,2),(2,3)}
    :return PINC score between sentence (a) and sentence (b)
    """
    sum = 0
    index = 0

    for i in range(1, n + 1):
        s = set(nltk.ngrams(input_utterance, i, pad_left=False, pad_right=False))
        p = set(nltk.ngrams(paraphrase, i, pad_left=False, pad_right=False))
        if s and p:
            index += 1
            intersection = s.intersection(p)
            sum += 1 - len(intersection) / len(p)

    if index == 0:
        return 0
    return sum / index


def jaccard_index(input_utterance: str, paraphrase: str, n=3) -> float:
    """
    Calculate the reverse of the mean Jaccard Index between the sentences’ n-grams sets to represent the semantic distances between the two sentences
    :param source: sentence a
    :param paraphrase: sentence b
    :param n: n-grams, default n=3 as set by author papers in their experiments e.g a=[1,2,3] b=[2,3,4] => 2-grams of a: {(1,2),(2,3)}
    :return reverse of the mean Jaccard Index between sentence a and sentence b
    """
    sum = 0
    for i in range(1, n + 1):
        s = set(nltk.ngrams(input_utterance, i, pad_left=False, pad_right=False))
        p = set(nltk.ngrams(paraphrase, i, pad_left=False, pad_right=False))
        if s and p:
            intersection = s.intersection(p)  # intersection between s and p
            p = s.union(p)  # union between s and p
            jaccard_index = len(intersection) / len(
                p
            )  # The Jaccard index, also known as Intersection over Union
            sum += jaccard_index
    # return reverse of the mean of Jaccard index
    return 1 - sum / n


def compute_div(dataset: Dict[str, List[str]]) -> float:
    """
    Compute diversity as the average jaccard_index distance between all sentence pairs from the paper
    "Data Collection for Dialogue System: A Startup Perspective"

    :param dataset: Dataset to be measured in terms of diversity.
                    Python dictionary, key: initial utterance - value: list of paraphrases
    :return Diversity score for the whole dataset.
    """

    total_d = 0
    d_list = []
    for expr in dataset:
        local_d = 0
        index = 0
        for i, ps in enumerate(dataset[expr]):
            tokens_1 = tokenize.word_tokenize(__pre_process(ps))
            tokens_1 = list(
                filter(None, tokens_1)
            )  # remove empty string from list of tokens and convert filter object to list
            for j, p in enumerate(dataset[expr]):
                if j != i:
                    tokens_2 = __pre_process(p).split(" ")
                    tokens_2 = list(filter(None, tokens_2))
                    local_d += jaccard_index(
                        tokens_1, tokens_2
                    )  # jaccard_index(tokens_1, tokens_2,4) # to compute 4-gram DIV
                    index += 1

        if index != 0:
            local_d = local_d / index
            d_list.append(local_d)
            total_d += local_d

    return total_d / len(dataset)


def compute_corpus_bleu(dataset: Dict[str, List[str]]) -> float:
    """
    Returns the average BLEU score the given dataset.

    :param dataset: Dataset to be measured in terms of BLEU.
                    Python dictionary, key: initial utterance - value: list of paraphrases
    :return BLEU score for the whole dataset.
    """
    score = 0
    smooth_fn = SmoothingFunction()

    for k, v in dataset.items():
        reference = k.lower().split(" ")
        utterance_bleu_score = 0
        for cand in v:
            candidate = cand.lower().split(" ")
            paraphrase_bleu_score = sentence_bleu(
                candidate, reference, smoothing_function=smooth_fn.method1
            )

            utterance_bleu_score += paraphrase_bleu_score

        if utterance_bleu_score > 0:
            utterance_bleu_score = utterance_bleu_score / len(v)
        score += utterance_bleu_score

    bleu = score / len(dataset)
    return bleu


def compute_corpus_gleu(dataset: Dict[str, List[str]]) -> float:
    """
    Returns the average GLEU score the given dataset.

    :param dataset: Dataset to be measured in terms of BLEU.
                    Python dictionary, key: initial utterance - value: list of paraphrases
    :return GLEU score for the whole dataset.
    """
    score = 0

    for k, v in dataset.items():
        reference = k.lower().split(" ")
        utterance_bleu_score = 0
        for cand in v:
            candidate = cand.lower().split(" ")
            paraphrase_bleu_score = sentence_gleu(candidate, reference)

            utterance_bleu_score += paraphrase_bleu_score

        if utterance_bleu_score > 0:
            utterance_bleu_score = utterance_bleu_score / len(v)
        score += utterance_bleu_score

    gleu = score / len(dataset)
    return gleu


# ------------- internal ------------
def __pre_process(text):
    # lemmatize and lowercase
    text = wordnet_lemmatize(text)
    return re.sub(r"[^\w\s]", " ", text).lower()
