import itertools
from copy import deepcopy

from symbols_table import *
from term import Term
import numbers
from functools import reduce
import operator

"""     These functions are really methods on term.
    They can modify the term.
    
        simplify() takes as argument ditionary with rules of simplification to apply to terms.
    Those simplifications are of form: def rule(term: Term, rules: dict) -> Term, bool. They take a term,
    make one simplification if its possible, and return possibly modified term and bool indicating if any simplification was made.
    Most of them don't need to take rules dict as a parameter - but there's one that needs it because it calls simplify() itself in its body.
    The fix I chose (as it emerged late) is: 
                                1. functions are expected to take rules as parameter so simplify just passes the same dict that it got.
                                2. decorator is used to artificially add new parameter to function definitions, 
                                   as there is no need to use it in most of them.
"""


def simplify(term1):
    term1 = preprocess(term1)
    term1 = do_simplifications(term1, rules_for_simplification)
    term1 = postprocess(term1)
    return term1


def is_positive_integer(x):
    if float(x).is_integer() and x > 0:
        return True
    return False


def preprocess(term1) -> Term:
    """ Change  1. x^n to x*x*..*x.
                2. substraction to addition with *(-1) """
    if term1.label == SUB:
        return Term(ADD, [preprocess(term1.children[0])] + [Term(MUL, [Term(-1), preprocess(subtrahend)]) for subtrahend in term1.children[1:]])
    elif term1.label == POW and is_positive_integer(term1.children[1].label):  # natural number power
        return Term(MUL, [preprocess(deepcopy(term1.children[0])) for _ in range(term1.children[1].label)])
    else:
        return Term(term1.label, [preprocess(child) for child in term1.children])


def postprocess(term1) -> Term:
    """ Change  1. x*x*..*x to x^n .
                2. addition with negative coefficients back to subtraction. """
    if term1.label == MUL and len(term1.children) >= 2 and terms_equal(*term1.children):
        return Term(POW, [postprocess(term1.children[0]), Term(len(term1.children))])
    elif term1.label == ADD:
        new_summands = [term1.children[0]]
        for summand in term1.children[1:]:
            if summand.label == MUL:
                for i, a in enumerate(summand.children):  # look for negative number
                    if isinstance(a.label, numbers.Number) and a.label < 0:  # if found change +(-a)*term to -a*term
                        a.label = abs(a.label)
                        if a.label == 1 and len(summand.children) > 1:
                            summand.children.pop(i)  # delete unnecessary 1 in mul
                        new_summands[-1] = Term(SUB, [new_summands[-1], summand])
                        break
                else:  # negative number not found
                    new_summands.append(summand)
            else:
                new_summands.append(summand)
        if len(new_summands) == 1:
            return new_summands[0]
        else:
            return Term(ADD, new_summands)
    else:
        return term1


def do_simplifications(term1: Term, rules) -> Term:
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
                term2, rule_changed = rule(term2, rules)
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


def terms_equal(term1: Term, term2: Term, *terms) -> bool:
    """ Takes minimal of 2 terms, returns True if all the terms have equal labels and equal children.
        Children are compared in order and need to be equal in order."""
    labels_equal = term1.label == term2.label and all(term1.label == term.label for term in terms)
    if labels_equal:
        lengths_equal = len(term1.children) == len(term2.children) and all(len(term1.children) == len(term.children) for term in terms)
        if lengths_equal:
            children_equal = all(terms_equal(child1, child2, *childs)
                                 for child1, child2, *childs in zip(term1.children, term2.children, *(map(lambda x: x.children, terms))))
            if children_equal:
                return True
    return False


def add_unused_parameter(f: callable):
    def foo(term, rules):
        return f(term)
    return foo


@add_unused_parameter
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


@add_unused_parameter
def single_term(term1: Term) -> (Term, bool):
    assert term1.label in [ADD, MUL]
    if len(term1.children) == 1:
        return term1.children[0], True
    return term1, False


@add_unused_parameter
def distributive_property(term1: Term) -> (Term, bool):
    """x*(1+3) -> x*1 + x*3"""
    if len(term1.children) == 2:
        left, right = term1.children[0], term1.children[1]
        if right.label == ADD:
            return Term(ADD, [Term(MUL, [left, addend]) for addend in right.children]), True
        elif left.label == ADD:
            return Term(ADD, [Term(MUL, [right, addend]) for addend in left.children]), True
    return term1, False


def perform_calculations_on_numbers_add_mul(term1: Term, operator) -> (Term, bool):
    """ generic function for add_numbers, multiply_numbers.
        it simplifies a term performing term operation on numbers among its children."""
    children = []
    result = None
    for child in term1.children:
        if isinstance(child.label, numbers.Number):
            if result is not None:
                result = operator(result, child.label)
            else:
                result = child.label
        else:
            children.append(child)

    if result is not None:
        children.append(Term(result))

    return Term(term1.label, children), len(children) < len(term1.children)


def perform_calculations_on_numbers_sub_div_pow(term1: Term, operator) -> (Term, bool):
    """ generic function for subtract_numbers, divide_numbers, raise_to_power_numbers.
        it simplifies a term performing term operation on numbers among its children.
        it is seperate to perform_calculations_on_numbers_1 as those operations
        expect binary term. """
    if len(term1.children) == 2:
        left, right = term1.children
        if isinstance(left.label, numbers.Number) and isinstance(right.label, numbers.Number):
            return Term(operator(left.label, right.label)), True
    return term1, False


@add_unused_parameter
def add_numbers(term1: Term) -> (Term, bool):
    assert term1.label == ADD
    """finds numbers in a sum and ads them.
        4+3+x+x -> 7+x+x"""
    return perform_calculations_on_numbers_add_mul(term1, operator.add)


@add_unused_parameter
def multiply_numbers(term1: Term) -> (Term, bool):
    assert term1.label == MUL
    """finds numbers in a sum and ads them.
        4*3*x*x -> 12*x*x"""
    return perform_calculations_on_numbers_add_mul(term1, operator.mul)


@add_unused_parameter
def subtract_numbers(term1: Term) -> (Term, bool):
    assert term1.label == SUB
    assert len(term1.children) == 2
    """subtracts numbers in a subtraction.
        4-3 -> 1"""
    return perform_calculations_on_numbers_sub_div_pow(term1, operator.sub)


@add_unused_parameter
def divide_numbers(term1: Term) -> (Term, bool):
    assert term1.label == DIV
    assert len(term1.children) == 2
    """subtracts numbers in a subtraction.
        3/4 -> 0.75"""
    return perform_calculations_on_numbers_sub_div_pow(term1, operator.truediv)


@add_unused_parameter
def raise_to_power_numbers(term1: Term) -> (Term, bool):
    assert term1.label == POW
    assert len(term1.children) == 2
    """subtracts numbers in a subtraction.
        3/4 -> 0.75"""
    return perform_calculations_on_numbers_sub_div_pow(term1, operator.truediv)


@add_unused_parameter
def one_zero_in_exponent(term1: Term) -> (Term, bool):
    basis, index = term1.children[0], term1.children[1]
    if index.label == 0:
        return Term(1), True
    if index.label == 1:
        return basis, True
    if basis.label == 1:
        return Term(1), True
    if basis.label == 0:
        return Term(0), True
    return term1, False


@add_unused_parameter
def zero_in_sum(term1: Term) -> (Term, bool):
    assert term1.label == ADD
    changed = False
    children = []
    for child in term1.children:
        if child.label == 0:  # then skip it
            changed = True
        else:
            children.append(child)
    """here we check whether sum becomes empty after this operation.
       If this will become the case, that there is other simplification
       that can change number of children to 0,
       then extract this as seperate simplification - (or don't actually, 
       some other function that for's on children may try to simplify this strangely)"""
    if not children:  # empty sum
        return Term(0), True
    else:
        return Term(ADD, children), changed


# copy paste from zero_in_sum
@add_unused_parameter
def one_in_product(term1: Term) -> (Term, bool):
    assert term1.label == MUL
    changed = False
    children = []
    for child in term1.children:
        if child.label == 1:  # then skip it
            changed = True
        else:
            children.append(child)
    """here we check whether sum becomes empty after this operation"""
    if not children:  # empty product
        return Term(1), True
    else:
        return Term(MUL, children), changed


@add_unused_parameter
def zero_in_product(term1: Term) -> (Term, bool):
    assert term1.label == MUL
    if any(0 == child.label for child in term1.children):
        return Term(0), True
    else:
        return term1, False


@add_unused_parameter
def one_in_denominator(term1: Term) -> (Term, bool):
    assert term1.label == DIV
    assert len(term1.children) == 2  # assert binary tree
    if term1.children[1].label == 1:
        return term1.children[0], True
    return term1, False


@add_unused_parameter
def zero_in_sub(term1: Term) -> (Term, bool):
    assert term1.label == SUB
    changed = False
    children = [term1.children[0]]
    for child in term1.children[1:]:  # tbh its right_child
        if child.label == 0:  # then skip it
            changed = True
        else:
            children.append(child)
    if len(children) == 1:
        return children[0], True  # term1.children[0]
    else:
        return Term(SUB, children), changed


def generate_index_pairs(term1):
    """ This function yields lists to be used in extract_term() and try_to_simplify().
        List format is a list of pairs (i, j or None) - pairs of 2 indexes or 1 index and None.
        Here term is a sum and we want to check if there is term that we can extract from the sum.
        For it we need to generate indexes of terms in this sum term to be later checked for equality.
        For example if term1 is 4*x+5*x+x and list is [(1, 0), (2, None)]
        this chooses first element in second product (5 from 5*x) and the only element from last 'product' (x from x).
        Precisely - let's treat summands in term1.children as sets of its children if its multiplication or 1 element set if its not;
        then a list of indexes is yield for each subset of summands,
        for each element of cartesian product of these summands."""
    lengths = []  # will keep number or terms in a product or None if not a product for each summand in term1
    for child in term1.children:
        if child.label == MUL:
            lengths.append(len(child.children))
        else:
            lengths.append(None)

    for r in range(2, len(lengths)+1):
        for comb in itertools.combinations(range(len(lengths)), r):
            AxBxC = []
            for i in comb:
                if lengths[i] is None:  # this summand is a single term - not a product
                    AxBxC.append([(i, None)])
                else:                   # this summand is a product - append all pairs (i, j) corresponding to j-th term in product
                    AxBxC.append([(i, j) for j in range(lengths[i])])
            for elem in itertools.product(*AxBxC):
                yield elem


def try_to_simplify(term1, index_pairs, outer_op, inner_op, neutral_element, rules):
    """ Function to be used inside extract_term.
        It takes a term and information on which terms are equal so can be extracted out of the sum.
        It does exactly that - extract the term out of the sum
        and tries to simplify() whats left. If it can simplify it does that and function
        returns extracted_term * whats_simplified among other summands of term1 that are left untouched.
        Further simpification back to the sum is left to distributive_property().
        If no simplification can be made term is untouched - function tries to simplify on deepcopy() of terms.

        index_pairs describe where are the equal terms - its a single list yield by generate_index_pairs.
        List format is a list of pairs (i, j or None) so pairs of 2 indexes or 1 index and None.
        First index corresponds to index in term1.children, second to index in child.children. 'None' means its child itself.
    """
    # need to make a deepcopy of term1.children
    # want to try simplifying, but reverse changes if can't simplify
    children_copy = deepcopy(term1.children)

    # take any term of those equal terms
    if index_pairs[0][1] is None:  # term is a simple term (i.e: x)
        extracted_term = children_copy[index_pairs[0][0]]
    else:                       # term is MUL term (i.e: (* x 4))
        extracted_term = (children_copy[index_pairs[0][0]]).children[index_pairs[0][1]]

    children_left_untouched = []
    children_after_term_extracted = []
    positions_dict = dict(index_pairs)  # child in term index :  subchild in child.chldren or None
    affected_indexes = set((i for (i, j) in index_pairs))
    for i, child in enumerate(children_copy):
        if i in affected_indexes:
            if positions_dict[i] is None:
                children_after_term_extracted.append(Term(1))
            else:
                j = positions_dict[i]
                child.children.pop(j)  # modifies child
                children_after_term_extracted.append(child)
        else:
            children_left_untouched.append(child)

    old_length = len(children_after_term_extracted)
    term_after_extracted = Term(ADD, children_after_term_extracted)
    simplified_term = do_simplifications(term_after_extracted, rules)
    simplified = len(simplified_term.children) < old_length or simplified_term.label != ADD  # did it do anything?

    if simplified:
        new_term = Term(MUL, [extracted_term, simplified_term])
        # ^ we rely on distributive property implemented in other function
        # to distribute extracted term back to terms.
        children_left_untouched.append(new_term)  # already touched
        return Term(ADD, children_left_untouched), True
    else:
        return None, False


def extract_term(term1: Term, rules: dict) -> (Term, bool):
    assert term1.label == ADD
    # if any(len(child.children) == 1 for child in term1.children if child.label == MUL):
    #     # extract term on term1 can trigger before single_term simplifies term1.children
    #     # leading to a bug where extract term extract single term and leaves MUL empty.
    #     return term1, False
    for index_pairs in generate_index_pairs(term1):
        terms_on_indexes = map(lambda pair: term1.children[pair[0]] if pair[1] is None else term1.children[pair[0]].children[pair[1]],
                               index_pairs)
        if terms_equal(*terms_on_indexes):  # here generate_index_pairs only yields index lists of length 2 or more
            new_term, simplified = try_to_simplify(term1, index_pairs, rules)
            if simplified:
                return new_term, simplified
    return term1, False


""" Rules are applied starting from the last one in the list."""
rules_for_simplification = {ADD: [extract_term, associativity, single_term, zero_in_sum, add_numbers],
                            MUL: [associativity, one_in_product, distributive_property, single_term, multiply_numbers, zero_in_product],
                            DIV: [one_in_denominator, divide_numbers],
                            SUB: [zero_in_sub, subtract_numbers],
                            POW: [one_zero_in_exponent, raise_to_power_numbers]}

if __name__ == "__main__":
    from lexer import tokenize
    from parse import Parser
    from printing import to_pretty_string
    # TODO whats below - preprocesing
    # TODO postprocessing
    # TODO printing
    """ !!!
        now should write all this stuff but for subtraction??
        no way - preprocess formula to change subtraction to addition,
        also x^n to iterated x*x*x.. (test this for big n)
    """
    # formula = "(x^2*5*6*1)*3-x^2"
    # tokens = tokenize(formula)
    # parser = Parser(tokens)
    # term1 = parser.parse()
    # simplified = do_simplifications(term1, rules_for_simplification)
    # print(formula)
    # print(simplified)
    # print(to_pretty_string(simplified))

    formula = "x*x-x*x*x*x"
    tokens = tokenize(formula)
    parser = Parser(tokens)
    term1 = parser.parse()
    simplified = simplify(term1)
    print(formula)
    print(simplified)
    print(to_pretty_string(simplified))


# def product_reduction(term1: Term) -> (Term, bool):
#     """ rule for multiplication
#         3 * 4 * x * x = 12 * (x ^ 2)"""
#     # print(term1)
#     nums = []
#     variable_count = 0
#     children = []
#
#     for child in term1.children:
#         if isinstance(child.label, numbers.Number):
#             nums.append(child.label)
#         elif child.label == VAR:
#             variable_count += 1
#         else:
#             children.append(child)
#
#     changed = False
#     if variable_count > 1:
#         children.append(Term(POW, [Term(VAR), Term(variable_count)]))
#         changed = True
#     elif variable_count == 1:
#         children.append(Term(VAR))
#     if nums:
#         product = reduce(lambda x, y: x * y, nums)
#         if product != 1:
#             children.append(Term(product))
#         if len(nums) > 1:
#             changed = True
#     if len(children) == 1:
#         return Term(children[0].label, children[0].children), True
#     else:
#         return Term(MUL, children), changed



# def sum_reduction(term1: Term) -> (Term, bool):
#     """5+4+x+x = 2*x+9"""
#     sum = 0
#     variable_count = 0
#     children = []
#     for child in term1.children:
#         if isinstance(child.label, numbers.Number):
#             sum += child.label
#         elif child.label == VAR:
#             variable_count += 1
#         else:
#             children.append(child)
#
#     if variable_count == 1:
#         children.append(Term(VAR))
#     elif variable_count > 1:
#         children.append(Term(MUL, [Term(variable_count), Term(VAR)]))
#     if sum != 0:
#         children.append(Term(sum))
#
#     changed = len(children) != len(term1.children)
#     if len(children) == 1:
#         return children[0], changed
#     else:
#         return Term(ADD, children), changed


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



""" More complex but better perhaps way to do all reductions would be:
    - reduction only adds/multiplies numbers
    - simplification of kind x+x-2*x = 0 achieved through finding common factor in sum
    - reduce subtraction to sum
    - think broadly - does not suffice to find x's in products and count them. 
    Maybe need to check if terms are equal.
    - so 4*x^2+x^2 = x^2*(4+1) = 5*x^2
    - probably complicated and unnecessary: x^2-x = x*(x-1)  (<- might be solved)
        note: don't simplify to x^n, make it in the end, first its probably easier to just stack x's like x*X*x
    """

""" its more complicated ^:
    - to simplify 2+2*x+3*x = 2+5*x needs not only to extract common factor for all summands
    but also try to extract something for each subsets of summands
    - tho it doesn't make sense to randomly add parentheses and chaos
    - possible solution: if common factor is found for a group of terms then do:
        1. store terms
        2. extract term
        3. simplify terms left
        4. if no simplification can be done return unchanged terms"""

"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    functions modify terms, doing term = Term(symbol, children=children) does not copy children obvs."""