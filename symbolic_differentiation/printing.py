from symbolic_differentiation.symbols_table import *
import numbers

""" Module exports to_pretty_string() which takes Term and returns the expression as a string with brackets etc. """

precedence = {ADD: 6, SUB: 6, MUL: 8, DIV: 8, POW: 10, SIN: 12, TAN: 12, COS: 12, LOG: 12}

# improvements to implement (maybe (probably not)) :
# 1. bracket negative numbers
# 2. put numbers in a product to the beginning


def to_string_with_precedence(term1, prec):
    if term1.label == VAR or isinstance(term1.label, numbers.Number):
        return str(term1.label)
    elif term1.label in [SIN, COS, TAN, LOG]:
        return term1.label + OPEN_BR + to_string_with_precedence(term1.children[0], 0) + CLOSE_BR
    elif term1.label in [ADD, MUL]:
        ret = to_string_with_precedence(term1.children[0], precedence[term1.label])
        for child in term1.children[1:]:
            ret += term1.label + to_string_with_precedence(child, precedence[term1.label])
        if precedence[term1.label] < prec:
            ret = OPEN_BR + ret + CLOSE_BR
        return ret
    elif term1.label in [SUB, DIV]:
        # here expect term1 to be binary.
        ret = (to_string_with_precedence(term1.children[0], precedence[term1.label])
               + term1.label + to_string_with_precedence(term1.children[1], precedence[term1.label] + 1))
        if precedence[term1.label] < prec:
            ret = OPEN_BR + ret + CLOSE_BR
        return ret
    elif term1.label == POW:
        ret = (to_string_with_precedence(term1.children[0], precedence[term1.label] + 1)
               + term1.label + to_string_with_precedence(term1.children[1], precedence[term1.label] + 1))
        if precedence[term1.label] < prec:
            ret = OPEN_BR + ret + CLOSE_BR
        return ret
    else:
        raise Exception(f"Unrecognized term - don't know how to show it. Term: {term1}")


def to_pretty_string(term1):
    return to_string_with_precedence(term1, 0)


if __name__ == "__main__":
    from symbolic_differentiation.term import Term
    term = Term(ADD, [Term(SUB, [Term(3), Term(4)]), Term(SUB, [Term(SUB, [Term(POW, [Term(VAR), Term(ADD, [Term(1), Term(0)])]), Term(VAR)]), Term(MUL, [Term(3), Term(VAR)])])])
    formula = to_pretty_string(term)
    print(formula)