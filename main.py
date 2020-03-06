import numbers
from functools import reduce
from symbols_table import *


class Term:

    def __init__(self, label: str or int, children=None):
        # self.type
        self.label = label
        # self.type = variable/number/function
        self.children = children if children else []

    def __repr__(self):
        if self.children:
            return f"({self.label} " + reduce(lambda str1, str2: str1+" "+str2, (repr(child) for child in self.children)) + ")"
        else:
            return f"{self.label}"

    def to_pretty_string(self) -> str:
        if self.label == ADD:
            return self.add_to_str()
        elif self.label in [MUL, DIV, SUB]:
            return self.mul_div_to_str(self.label)
        elif self.label in [SIN, COS, TAN, LOG]:
            return self.label + OPEN_BR + self.children[0].to_pretty_string() + CLOSE_BR
        elif self.label == VAR or isinstance(self.label, numbers.Number):
            return str(self.label)

    def add_to_str(self):
        return reduce(lambda str1, str2: str1+"+"+str2, (child.to_pretty_string() for child in self.children))

    def mul_div_to_str(self, OP):
        # start with first
        if self.children[0].label in [ADD, SUB]:
            txt = OPEN_BR + self.children[0].to_pretty_string() + CLOSE_BR
        else:
            txt = self.children[0].to_pretty_string()

        for child in self.children[1:]:
            # add brackets for sum and sub
            if child.label in [ADD, SUB]:
                txt += OP + OPEN_BR + child.to_pretty_string() + CLOSE_BR
            else:
                txt += OP + child.to_pretty_string()
        return txt

    def back_to_string(self):
        pass


class RulesDict:

    __key_for_variables = lambda:None  # some unique crap to be used as a key

    def __init__(self, rules, variables):
        self.rules = rules
        self.variables = set(variables)

    def __getitem__(self, item):
        if isinstance(item, numbers.Number):
            return self.rules[numbers.Number]
        elif item in self.variables:
            return self.rules[RulesDict.__key_for_variables]
        else:
            return self.rules[item]

# a note - probably i want simple dict with rules for operations, numbers and variables defined in differentiate


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


def associativity(term1: Term) -> (Term, bool):
    """assumes careful use = term1.label is a functional symbol of associative function
    e.g. takes (* (* x 2) 3) returns (* x 2 3)"""

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
    else: return term1, False


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
        product = reduce(lambda x, y: x*y, nums)
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
        if isinstance(term1.label, numbers.Number):
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


rules_for_simplification = {MUL: [associativity, zero_in_product, product_reduction],
                            ADD: [associativity, zero_in_sum, sum_reduction],
                            SUB: [zero_in_sub]}

if "__main__" == __name__:
    from derivatives import derivative, rules_for_differentiation
    from lexer import tokenize
    from parse import Parser
    formula = "log(x+4)+x*sin(x)"
    tokens = tokenize(formula)
    parser = Parser(tokens)
    term1 = parser.parse()
    deriv_term1 = derivative(term1, rules_for_differentiation)
    simplified_deriv = simplify(deriv_term1, rules_for_simplification)

    # TODO SUB (-) parsed wrong!!! Can only fix printing if parsing fixed.

    pprint_formula = 'x-x/4-x'
    pprint_tokens = tokenize(pprint_formula)
    parser1 = Parser(pprint_tokens)
    pprint_term = parser1.parse()
    print(pprint_term)
    print(pprint_term.to_pretty_string())
