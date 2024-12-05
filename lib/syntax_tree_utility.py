from typing import TypeVar,List
import re

from nltk.tree import Tree
from nltk import ParentedTree
import numpy as np
from nltk.tree import TreePrettyPrinter
from zss import simple_distance
from zss import Node

""" 
    This module calculates the tree edit distance between the constituency parse trees after removing the keywords between a pair of syntax trees
    Inspired from this paper https://aclanthology.org/P19-1599/
    the code of the functions build_tree() strdist() and compute_tree_edit_distance() has been copied from the following git repo: https://github.com/mingdachen/syntactic-template-generation/blob/master/eval_utils.py

    Activate the SyntacticPipeline to execute the script
"""

T1 = TypeVar('numpy.float64')#Type hints annotation for the return value of the compute_tree_edit_distance() function
T2 = TypeVar('zss.simple_tree.Node')#Type hints annotation for the return value of the build_tree() function


def is_parent(tok):
    """
    Check if a token represents a parent node in a constituency parse tree.

    :args
        tok (str): The token to check.

    :returns
        bool: True if the token represents a parent node (either '(' or ')'), False otherwise.
    
    :Source
        This function is adapted from the `utils.py` file in the GitHub repository: https://github.com/uclanlp/synpg/blob/master/utils.py
    """
    return tok == ")" or tok == "("

def deleaf(tree: ParentedTree) -> List[str]:
    """
    Removes leaf nodes (i.e. words) from the given NLTK.ParentedTree tree.

    :args
        tree: A NLTK.ParentedTree object representing the parse tree to remove leaf nodes from.

    :returns
        A list of non-leaf tokens after removing the leaf nodes.

    :source
        This function is adapted from the `scpn_utils.py` file in the GitHub repository: https://github.com/miyyer/scpn/blob/master/scpn_utils.py
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
            if not is_parent(tok1) and not is_parent(tok2):
                arr[n + 1] = ""

    nonleaves = " ".join(arr)
    return nonleaves.split()

def parse_tree_level_dropout(tree: ParentedTree, level=2) -> None:
    """
    Remove levels of a parse tree below a specific level or randomly select levels to construct a template.

    :args
        tree (ParentedTree): The parse tree to modify.
        level (int, optional): The specific level to drop from the parse tree. The level to drop from the full constituency parse tree. If not specified, a random
            level will be selected based on treerate. Defaults to None.

    :returns
        None: The function modifies the input `tree` in place.
    
    :Source
        This function is adapted from the `scpn_utils.py` file in the GitHub repository:
        https://github.com/miyyer/scpn/blob/master/scpn_utils.py

    :Note
        The function uses the `ParentedTree` class from the `nltk.tree` module.
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

def is_syntax_template_valid(syntax_template):
    """
    Check if a given syntax template is well-formed. Call this function before build_tree to avoid the following error when we have poorly constructed syntax tree.
    Error to avoid: ValueError: Tree.read(): expected 'end-of-string' but got ')'

    A syntax template is considered well-formed if the number of opening parentheses "("
    matches the number of closing parentheses ")".

    :args
        syntax_template (str): The syntax template to check.

    :return
        bool: True if the syntax template is well-formed, False otherwise.

    :Example
    >>> is_syntax_template_valid("(ROOT (NP (NNP) (NNP)))")
    True

    >>> is_syntax_template_valid("(ROOT (NP (NNP) (NNP))")
    False

    >>> is_syntax_template_valid("(ROOT (NP (NNP) (NNP))) )")
    False
    """
    stack = []
    for char in syntax_template:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return False
            stack.pop()
    return len(stack) == 0


def build_tree(s:str) -> T2:
    """
    Converts a string into a parse tree using the NLTK Tree module so that it is compatible with the zss input format.

    :args
        s (str): The syntax tree template/pattern as a string.

    :returns
        zss.simple_tree.Node: A syntax tree compatible with zss in the form of zss.simple_tree.Node.

    :source
        This function is adapted from the `build_tree` function in the `eval_utils.py` file of the GitHub repository:
        https://github.com/mingdachen/syntactic-template-generation/blob/master/eval_utils.py

    :Note
        The function uses the `Tree` and `Node` classes from the `nltk.tree` and `zss.simple_tree` modules, respectively.
        The `T2` type hint suggests the expected type of the returned value, but it should be replaced with the correct type.

    :Example
        Convert the syntax tree template `s` to a compatible zss tree format:
        if s="(ROOT (S (SBAR (WHNP (WP Whatever)) (S (VP (VBZ is) (VP (VBN forbidden))))) (VP (VBZ is) (VP (VBN desired)))))",
        then the converted tree would look like:
            1:S
            2:ROOT
            2:SBAR
            1:WHNP
            0:WP
            2:VP
            0:VBZ
            1:VP
            0:VBN
            2:VP
            0:VBZ
            1:VP
            0:VBN
    """

    old_t = Tree.fromstring(s)
    new_t = Node("S")

    def create_tree(curr_t, t):
        if t.label() and t.label() != "S":
            new_t = Node(t.label())
            curr_t.addkid(new_t)
        else:
            new_t = curr_t
        for i in t:
            if isinstance(i, Tree):
                create_tree(new_t, i)
    create_tree(new_t, old_t)
    return new_t

def strdist(a:str, b:str) -> 0|1:
    """
    A helper function for the label_distance parameter of the zss.simple_distance() function.
    Calculates the string distance between two labels, a and b, in a constituency parse tree.

    :args
        a (str): The label of nodeX in a constituency parse tree. E.g. a= 'V' or a='S' or a='VP'
        b (str): The label of nodeY in a constituency parse tree. E.g. b= 'S' or b='DT' or b='VP' or b='NP'
        Namely, a and b are tags contained in each node of a consistency analysis tree, eg. (S (NP (DT )) (VP (VBZ ) (ADJP (JJ ))) (. ))

    :returns
        int: The string distance between the labels a and b. Returns 0 if the labels are the same; otherwise, returns 1. By default, it is the string editing distance (if available);
        a number N representing the amount of changes it takes to turn one label into the other. Here we have relaxed values where 0 indicates that the labels are the same, otherwise 1.

    :details
        This function calculates the number of edits required to transform label a into label b.
        The string distance represents the difference between the labels based on the number of changes needed.
        The function returns 0 if the labels are the same, and 1 if they are different.

    For more details:
        See the documentation for zss.simple_distance(): https://zhang-shasha.readthedocs.io/en/latest/#zss.simple_distance
    """

    if a == b:
        return 0
    else:
        return 1

def compute_tree_edit_distance(pred_parse:str, ref_parse:str) -> T1:
    """
    A wrapper function to measure syntactic similarity by computing the Tree Edit Distance between a pair of syntax trees using the Zhang-Shasha (zss) library.

    :args
        pred_parse (str): The constituency parse tree of a candidate sentence as a string.
        ref_parse (str): The constituency parse tree of a reference template as a string.

    :returns
        numpy.float64: The Tree Edit Distance between the constituency parse trees as a numpy.float64 object.

    :source
        This function is adapted from the `eval_utils.py` file in the GitHub repository:
        https://github.com/mingdachen/syntactic-template-generation/blob/master/eval_utils.py

    :Note
        The function relies on the `build_tree` and `strdist` functions for tree conversion and label distance calculation,
        respectively.
    """

    """
        A wrapper function to measure the syntactic similarity by computing the Tree Edit Distance between a pair of syntax tree using the Zhang-Shasha (zss) library.
        The funtion get a pair of Syntax tree(pred_pasre and ref_parse) which are respectively the syntax parse tree of a candidate sentence that have to be compared with a reference template.
        Source code of the function: https://github.com/mingdachen/syntactic-template-generation/blob/master/eval_utils.py

        args:
            pred_parse: a string consituency parse tree of a contidate. E.g. (SBARQ (ADVP) (,) (S) (,) (SQ))
            ref_parse: a string consituency parse tree. E.g. (SBARQ (ADVP) (,) (S) (,) (SQ))
        
        return:
            The Tree Edit Distance between constituency parse trees as numpy.float64 object
        
    """
    return simple_distance(  build_tree(pred_parse), build_tree(ref_parse), label_dist=strdist)


"""
    Obtain a parse template as defined in "Adversarial Example Generation with Syntactically Controlled Paraphrase Networks".
    Basically this function returns the top two levels of the constituency parse tree.
"""

def get_parse_template(parse_string: str, level:int = 2) ->str:
    """
    Extracts a syntax parse template from a given constituency parse string by dropping levels of the parse tree.
    The definition of a syntax template is adopted from Iyyer et al.[1], where it refers to the top two levels of a constituency parse tree.
    [1] Iyyer, M., Wieting, J., Gimpel, K., & Zettlemoyer, L. (2018). Adversarial example generation with syntactically controlled paraphrase networks. arXiv preprint arXiv:1804.06059.

    :args
        parse_string (str): A string representation of the parse tree.
        level (int, optional): The level at which to drop constituents from the parse tree (default: 2).

    :returns
        str: A parse template string with leaf nodes (words) removed.

    :Note
        The function uses the `parse_tree_level_dropout` and `deleaf` functions internally to drop levels and remove leaf nodes, respectively.

    :Example
        >>> parse_string = "(ROOT (S (NP (DT The) (NN cat)) (VP (VBD sat) (PP (IN on) (NP (DT the) (NN mat))))))"
        >>> template = get_parse_template(parse_string, level=3)
        >>> print(template)
        "(ROOT (S (NP) (VP (VBD) (PP (IN) (NP (DT) (NN))))))"

    """
    parsed_tree = ParentedTree.fromstring(parse_string)
    parse_tree_level_dropout(parsed_tree, level=level)
    template = deleaf(parsed_tree)
    template = " ".join(template)#convert the deleted parse list to string in onrder to compute later the Tree Edit Distance
    
    return template

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

    # Remove the 'ROOT' label occurence
    phrases_to_remove = ["ROOT", " ROOT", " ROOT ", "ROOT "]
    for phrase in phrases_to_remove:
        cleaned_tree_str = cleaned_tree_str.replace(phrase, '')

    return cleaned_tree_str.strip()

### Test ###
def draft_test():
    """
    A draft function to learn how to use nltk's ParentedTree and TreePrettyPrinter, and how to delete leaves using the deleaf function.
    This function demonstrates the usage of nltk's `ParentedTree` and `TreePrettyPrinter`, as well as the deletion of leaves using the `deleaf` function.

    1. It defines two constituency parse trees `a` and `b` as strings.
    2. It computes the Tree Edit Distance (TED) between `a` and `b` using the `compute_tree_edit_distance` function.
    3. It prints the computed TED and displays a message based on whether the TED is 0 or not.
    4. It modifies `a` by removing the leaf nodes using the `deleaf` function and joining the modified tokens.
    5. It creates a `ParentedTree` object `a_tree` from the modified `a` string.
    6. It computes the height of `a_tree` and prints it.
    7. It displays the original parse tree `a_tree` using the `TreePrettyPrinter`.
    8. It applies the `parse_tree_level_dropout` function to drop nodes from `a_tree` at level 2.
    9. It displays the modified parse tree `a_tree` after dropping nodes at level 2.
    10. It prints the string representation of `a_tree`.

    :returns
        None
    """
    a = "(ROOT (S (SBAR (WHNP (WP Whatever)) (S (VP (VBZ is) (VP (VBN forbidden))))) (VP (VBZ is) (VP (VBN desired)))))"
    b = "(S (SBAR (IN If) (S (NP (NN tomorrow)) (VP (VBZ is) (NP (DT a) (JJ sunny) (NN day))))) (, ,) (NP (PRP we)) (VP (MD will) (VP (VB have) (NP (DT a) (NN picnic)))) (. .))"

    c = compute_tree_edit_distance(a,b)
    print(f"After:{type(c)}\n{c}")
    if c == 0:
        print("Ca farte")
    else:
        print("Ca ne farte pas")
    
    a = "(SQ (MD  ) (MD Could) (NP (PRP you)) (VP (VB point) (NP (PRP me)) (PP (IN towards) (NP (NP (DT a) (NN store)) (SBAR (WHNP (WDT that)) (S (VP (VBZ sells) (NP (JJ fresh) (NN meat)))))))) (. ?))"
    a = deleaf(a)
    a = " ".join(a)
    print(f"A: {a}\n===========\n")

    a = "(S (VP (VB Get) (NP (DT all) (NNS records) (PP (IN of) (NP (NNS customers))))) (SBAR (WHNP (WDT whose)) (S (NP (NN purchase) (NN amount)) (VP (VBZ exceeds) (NP (CD $) (CD 100))))))"
    a_tree = ParentedTree.fromstring(a)
    h = a_tree.height()
    print(f"\nHeight: {h}\n")
    print(f"\n===== Before Drop ======\n")
    print(TreePrettyPrinter(a_tree))
    # tree_dropout(a_tree,1,3)
    parse_tree_level_dropout(a_tree,1,2)
    print(f"\n======= After Drop level 3====\n")
    print(TreePrettyPrinter(a_tree))
    print(a_tree)
    print(f"The tree:\n{a_tree}")
    a = "(S (VP (VB Get) (NP (DT all) (NNS records) (PP (IN of) (NP (NNS customers))))) (SBAR (WHNP (WDT whose)) (S (NP (NN purchase) (NN amount)) (VP (VBZ exceeds) (NP (CD $) (CD 100))))))"

def ted_test():
    """
    A test function to compute the Tree Edit Distance using the zss simple_distance library.

    This function performs several operations to demonstrate the usage of the Tree Edit Distance (TED) computation:

    1. It defines a constituency parse tree `a` as a string.
    2. It removes the leaf nodes from `a` using the `deleaf` function.
    3. It prints and joins the modified `a` string.
    4. It creates a `ParentedTree` object `b` and a `Tree` object `a1` from the modified `a` string.
    5. It computes the height of `a1` and `b`.
    6. It displays the original parse tree `a1` and `b` before dropping nodes.
    7. It applies the `parse_tree_level_dropout` function to drop nodes from `b` at level 3.
    8. It displays the modified parse tree `b` after dropping nodes at level 3.
    9. It converts `b` to a string.
    10. It computes the Tree Edit Distance (TED) between `b` and itself using the `compute_tree_edit_distance` function.
    11. It prints the computed TED.

    :returns
        None.
    """

    a = "(S (VP (VB Get) (NP (DT all) (NNS records) (PP (IN of) (NP (NNS customers))))) (SBAR (WHNP (WDT whose)) (S (NP (NN purchase) (NN amount)) (VP (VBZ exceeds) (NP (CD $) (CD 100))))))"

    #first delete leaves
    a = deleaf(a)
    print(a)
    a = " ".join(a)

    print()
    l = len(a)+10
    print(" tree a without node: ".center(l,'-'))
    print(f" {a} ".center(l))
    print("-"*l)
    print()

    b = ParentedTree.fromstring(a)
    a1 = Tree.fromstring(a)
    ha = a1.height()
    h = b.height()
    print(f" Ha: {ha} - hb: {h} ".center(20,'-'))

    print()
    print(f" A Before Drop ".center(30,'-'))
    print(TreePrettyPrinter(a1))
    print(a1)
    pf = a1._pformat_flat("", "()", False)#linearize the tree
    print(f" {pf} ".center(100,'*'))

    print()
    print(f" B Before Drop ".center(30,'-'))
    print(TreePrettyPrinter(b))
    print(b)
    # tree_dropout(a_tree,1,3)
    parse_tree_level_dropout(b,3)

    print()
    print(f" B After Drop level 3 ".center(30,'-'))
    h = b.height()

    print()
    print(f"hb after drop: {h} ".center(30,'-'))
    print(TreePrettyPrinter(b))
    print(f" b = {b} ".center(100,'*'))

    print()
    b = str(b)#mandatory convert ParentedTree to string to be able to compute ted
    c = compute_tree_edit_distance(b,b)
    print(f" Tree Edit Distance(b,b)= {c} ".center(40,'-'))

def test_dropout_functions():
    """
    Test function for the tree_droparse_tree_level_dropout function.
    """
    # Example parse tree
    parse_tree = "(S (VP (VB Get) (NP (DT all) (NNS records) (PP (IN of) (NP (NNS customers))))) (SBAR (WHNP (WDT whose)) (S (NP (NN purchase) (NN amount)) (VP (VBZ exceeds) (NP (CD $) (CD 100))))))"

    # Drop constituents using parse_tree_level_dropout
    parsed_tree = ParentedTree.fromstring(parse_tree)
    print(TreePrettyPrinter(parsed_tree))
    parse_tree_level_dropout(parsed_tree, level=2)
    print(TreePrettyPrinter(parsed_tree))
    template = deleaf(parsed_tree)
    print(type(template))
    print(template)
    t = " ".join(template)
    print(t)
    c = compute_tree_edit_distance(t,t)
    print(f" Tree Edit Distance(b,b)= {c} ".center(40,'-'))

if __name__ == "__main__":
    test_dropout_functions()
