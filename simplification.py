from symbols_table import *
from term import Term
import numbers
from functools import reduce


def simplify(term1: Term, rules) -> Term:
    def simplify_helper(term2) -> (Term, bool):
        simplify_changed = False

        if term2.label in rules:
            '''if there are rules to apply,
            store them in list and apply one by one until term changes its label 
            and thus there are different rules to apply from now on.
            We apply rules while there are some rules for label.'''
            initial_term_label = term2.label
            list_of_rules = rules[term2.label].copy()
            while term2.label == initial_term_label and list_of_rules:
                rule = list_of_rules.pop()
                term2, rule_changed = rule(term2)
                simplify_changed = simplify_changed or rule_changed

        children = []
        for child in term2.children:
            simplified_child, rule_changed = simplify_helper(child)
            simplify_changed = simplify_changed or rule_changed
            children.append(simplified_child)

        return Term(term2.label, children), simplify_changed

    changing = True
    while changing:
        term1, simplify_changed = simplify_helper(term1)
        changing = changing and simplify_changed

    return term1


def terms_equal(term1: Term, term2: Term) -> bool:
    return (term1.label == term2.label and len(term1.children) == len(term2.children)
            and all(terms_equal(child1, child2) for child1, child2 in zip(term1.children, term2.children)))


def associativity(term1: Term) -> (Term, bool):
    """takes (* (* x 2) 3) returns (* x 2 3)"""

    children = []
    changed = False
    for child in term1.children:
        if child.label == term1.label:
            children.extend(child.children)
            changed = True
        else:
            children.append(child)

    return Term(term1.label, children), changed


def zero_in_product(term1: Term) -> (Term, bool):
    assert term1.label == MUL
    if any(0 == child.label for child in term1.children):
        return Term(0), True
    else:
        return term1, False


def product_reduction(term1: Term) -> (Term, bool):
    """ rule for multiplication
        3 * 4 * x * x = 12 * (x ^ 2)"""
    # print(term1)
    nums = []
    variable_count = 0
    children = []

    for child in term1.children:
        if isinstance(child.label, numbers.Number):
            nums.append(child.label)
        elif child.label == VAR:
            variable_count += 1
        else:
            children.append(child)

    changed = False
    if variable_count > 1:
        children.append(Term(POW, [Term(VAR), Term(variable_count)]))
        changed = True
    elif variable_count == 1:
        children.append(Term(VAR))
    if nums:
        product = reduce(lambda x, y: x * y, nums)
        if product != 1:
            children.append(Term(product))
        if len(nums) > 1:
            changed = True
    if len(children) == 1:
        return Term(children[0].label, children[0].children), True
    else:
        return Term(MUL, children), changed


def zero_in_sum(term1: Term) -> (Term, bool):
    assert term1.label == ADD
    changed = False
    children = []
    for child in term1.children:
        if child.label == 0:  # then skip it
            changed = True
        else:
            children.append(child)
    return Term(ADD, children), changed


def zero_in_sub(term1: Term) -> (Term, bool):
    assert term1.label == SUB
    changed = False
    children = [term1.children[0]]
    for child in term1.children[1:]:  # tbh its right_child
        if child.label == 0:  # then skip it
            changed = True
        else:
            children.append(child)
    return Term(SUB, children), changed


def sum_reduction(term1: Term) -> (Term, bool):
    """5+4+x+x = 2*x+9"""
    sum = 0
    variable_count = 0
    children = []
    for child in term1.children:
        if isinstance(child.label, numbers.Number):
            sum += child.label
        elif child.label == VAR:
            variable_count += 1
        else:
            children.append(child)

    if variable_count == 1:
        children.append(Term(VAR))
    elif variable_count > 1:
        children.append(Term(MUL, [Term(variable_count), Term(VAR)]))
    if sum != 0:
        children.append(Term(sum))

    changed = len(children) != len(term1.children)
    if len(children) == 1:
        return children[0], changed
    else:
        return Term(ADD, children), changed


# def subtraction_reduction(term1: Term) -> (Term, bool):
#     """4-x-3-x = 1-2*x
#     sub_reduction - subtraction :D"""
#     # note: this may be overcomplicated as SUB should always be binary
#     numbers_total = 0
#     variable_count = 0
#     children = []
#     # treat first element seperately
#     if
#     # loop will go for second element only
#     # but let it be a loop as i might change it later
#     for child in term1.children[1:]:
#         if isinstance(child.label, numbers.Number):
#             numbers_total -= child.label
#         #  ! i may want to add rule for
#         # - (+ a b) -> -a+-b
#         elif child.label == VAR:
#             variable_count -= 1
#         else:
#             children.append(child)


rules_for_simplification = {MUL: [associativity, zero_in_product, product_reduction], ADD: [associativity, zero_in_sum, sum_reduction],
                            SUB: [zero_in_sub]}  # , subtraction_reduction

""" More complex but better perhaps way to do all reductions would be:
    - reduction only adds/multiplies numbers
    - simplification of kind x+x-2*x = 0 achieved through finding common factor in sum
    - reduce subtraction to sum
    - think broadly - does not suffice to find x's in products and count them. 
    Maybe need to check if terms are equal.
    - so 4*x^2+x^2 = x^2*(4+1) = 5*x^2
    - probably complicated and unnecessary: x^2-x = x*(x-1)
    """
