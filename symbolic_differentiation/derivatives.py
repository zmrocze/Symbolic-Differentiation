from term import Term
from symbols_table import *
import numbers

""" derivative rules expect a term to have 2 children with binary operation (e.g. *) and one when unary (e.g. sin) (obviously xd)
    ! but differentiations and simplifications return various number of children, 
    ! so that simplifications cannot be done in between different differentiation steps"""


# rules should be a dict that is defined below these function definitions
def calculate_derivative(term1: Term, rules: dict) -> Term:
    """uses a rule which then loops to differentiate"""
    if isinstance(term1.label, numbers.Number):
        return Term(0)
    elif term1.label == VAR:
        return Term(1)
    else:
        return rules[term1.label](term1, rules)


def mul_derivative(term1: Term, rules) -> Term:
    """rule for differentiating a product:
    (uv)' = u'v + uv'"""
    child1, child2 = term1.children  # assumes binary tree
    return Term(ADD, [Term(MUL, [calculate_derivative(child1, rules), child2]), Term(MUL, [child1, calculate_derivative(child2, rules)])])


def div_derivative(term1: Term, rules) -> Term:
    """rule for differentiating division:
    (u/v)' = ((u')*v - u*(v'))/v^2"""
    child1, child2 = term1.children  # assumes binary tree
    return Term(DIV, [Term(SUB, [Term(MUL, [calculate_derivative(child1, rules), child2]), Term(MUL, [child1, calculate_derivative(child2, rules)])]),
                      Term(POW, [child2, Term(2)])])


def sum_derivative(term1: Term, rules) -> Term:
    """rule for differentiating sum and subtraction:
    (u+v)' = u' + v'"""
    child1, child2 = term1.children  # assumes binary tree
    return Term(term1.label, [calculate_derivative(child1, rules), calculate_derivative(child2, rules)])


def power_derivative(term1: Term, rules) -> Term:
    """rule for differentiating a power (hardcore):
    (u^v)' = v'*log(u)*u^v + v*u'*u^(v-1)"""
    child1, child2 = term1.children  # assumes binary tree
    return Term(ADD,
                [Term(MUL, [calculate_derivative(child2, rules), Term(LOG, [child1]), Term(POW, [child1, child2])]),
                 Term(MUL, [calculate_derivative(child1, rules), child2, Term(POW, [child1, Term(SUB, [child2, Term(1)])])])])


def sin_derivative(term1: Term, rules) -> Term:
    """rule for differentiating sinus:
    (sin(u))' = cos(u)*u'"""
    return Term(MUL, [Term(COS, [term1.children[0]]), calculate_derivative(term1.children[0], rules)])


def cos_derivative(term1: Term, rules) -> Term:
    """rule for differentiating cosinus:
    (cos(u))' = -sin(u)*u'  """
    return Term(MUL, [Term(-1), Term(SIN, [term1.children[0]]), calculate_derivative(term1.children[0], rules)])


def tan_derivative(term1: Term, rules) -> Term:
    """rule for differentiating tangens:
    (tan(u))' = u' + u'*(tan(u))^2"""
    arg = Term.copy(term1.children[0])
    return Term(ADD, [calculate_derivative(arg, rules), Term(MUL, [calculate_derivative(arg, rules), Term(POW, [term1, Term(2)])])])


def log_derivative(term1: Term, rules) -> Term:
    """rule for differentiating logarithm:
    (log(u))' = u'/u"""
    return Term(DIV, [calculate_derivative(term1.children[0], rules), term1.children[0]])


rules_for_differentiation = {MUL: mul_derivative, ADD: sum_derivative, SUB: sum_derivative,
                             COS: cos_derivative, SIN: sin_derivative, TAN: tan_derivative,
                             POW: power_derivative, DIV: div_derivative, LOG: log_derivative}


def derivative(term1):
    return calculate_derivative(term1, rules_for_differentiation)