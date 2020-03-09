from term import Term
from symbols_table import *
from functools import reduce
import numbers


def to_pretty_string(term1: Term) -> str:
    if term1.label == ADD:
        return sum_to_str(term1)
    elif term1.label == SUB:
        return sub_to_str(term1)
    elif term1.label in [MUL, DIV]:
        return mul_div_to_str(term1, term1.label)
    elif term1.label == POW:
        return pow_to_str(term1)
    elif term1.label in [SIN, COS, TAN, LOG]:
        return term1.label + OPEN_BR + to_pretty_string(term1.children[0]) + CLOSE_BR
    elif term1.label == VAR or isinstance(term1.label, numbers.Number):
        return str(term1.label)


def sum_to_str(term1: Term) -> str:
    return reduce(lambda str1, str2: str1+"+"+str2, (to_pretty_string(child) for child in term1.children))


def sub_to_str(term1: Term) -> str:
    """prints prints terms separated by - and adds
    brackets around sum."""
    txt = to_pretty_string(term1.children[0])

    for child in term1.children[1:]:
        # add brackets for sum and sub
        if child.label == ADD:
            txt += SUB + OPEN_BR + to_pretty_string(child) + CLOSE_BR
        else:
            txt += SUB + to_pretty_string(child)
    return txt


def mul_div_to_str(term1: Term, OP) -> str:
    """prints terms separated by * and adds
    brackets around sum and subtraction."""
    if term1.children[0].label in [ADD, SUB]:
        txt = OPEN_BR + to_pretty_string(term1.children[0]) + CLOSE_BR
    else:
        txt = to_pretty_string(term1.children[0])

    for child in term1.children[1:]:
        # add brackets for sum and sub
        if child.label in [ADD, SUB]:
            txt += OP + OPEN_BR + to_pretty_string(child) + CLOSE_BR
        else:
            txt += OP + to_pretty_string(child)
    return txt


def pow_to_str(term1: Term) -> str:
    """prints terms separated by ^ and adds
    brackets around sum, subtraction, multiplication and division."""
    if term1.children[0].label in [ADD, SUB, DIV, MUL]:
        txt = OPEN_BR + to_pretty_string(term1.children[0]) + CLOSE_BR
    else:
        txt = to_pretty_string(term1.children[0])

    for child in term1.children[1:]:
        # add brackets for sum and sub
        if child.label in [ADD, SUB, DIV, MUL]:
            txt += "^" + OPEN_BR + to_pretty_string(child) + CLOSE_BR
        else:
            txt += "^" + to_pretty_string(child)
    return txt
