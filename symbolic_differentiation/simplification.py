import operator
import numbers

from symbolic_differentiation.symbols_table import *
from symbolic_differentiation.term import Term

""" A simple expression differentiated yields huge expression that needs to be simplified e.g. D(x/x) = (1*x-x*1)/x^2
    
    Simplification is accomplished using many rules responsible for a single simplification.
    Rules are of type:  r :: (Term, dict) -> (Term, bool). They take a term, simplify it if they can 
    and return possibly modified term with boolean saying if any simplification was made.
    Second parameter is a dictionary with simplification rules - only one simplification rule really needs it so to most of them this paramater is added 
    artificially with add_unused_parameter decorator. 
    example rule:  zero_in_product(3*0*x^2, rules) = 0, True
    
    Simplification is run in a loop untill no more simplification can be made. (do_simplifications)
    One run of the loop applies rules recursively on terms, starting from the root, then children. (simplify_helper)
    Those functions record if a change was made.

    Before do_simplifications() is run, in the expression we change 
    subtraction to addition with the opposite expression and division to multiplication by inverse. (preprocess())
    In the end we change it back. (postprocess)
    
    This is packed into simplify() function that takes term and returns term simplified.
    
"""


def simplify(term1: Term) -> Term:
    """ Returns term simplified with arithmetic rules. """
    term1 = preprocess(term1)
    term1 = do_simplifications(term1, rules_for_simplification)
    term1 = postprocess(term1)
    return term1


def simplify_helper(term1, rules) -> (Term, bool):
    """ Function tries to apply simplifications and keeps track if any changes were made.
        Returns pair: modified term, bool if change was made.
        It applies simplifications top to bottom - first to the term1 and later to its children. """
    simplify_changed = False

    if term1.label in rules:
        '''if there are rules to apply,
        store them in list and apply one by one until term changes its label 
        and thus there are different rules to apply from now on.
        We apply rules while there are some rules for label.'''
        initial_term_label = term1.label
        list_of_rules = rules[term1.label].copy()
        while term1.label == initial_term_label and list_of_rules:
            rule = list_of_rules.pop()
            term1, rule_changed = rule(term1, rules)
            simplify_changed = simplify_changed or rule_changed

    children = []
    for child in term1.children:
        simplified_child, rule_changed = simplify_helper(child, rules)
        simplify_changed = simplify_changed or rule_changed
        children.append(simplified_child)

    return Term(term1.label, children), simplify_changed


def do_simplifications(term1: Term, rules) -> Term:
    """ Runs simplify_helper in a loop till no more simplifications can be applied.
        simplify_helper applies simplification rules - if there can be a change made it will do it, but it will not apply changes exhaustively.
    """
    changing = True
    while changing:
        term1, simplify_changed = simplify_helper(term1, rules)
        changing = changing and simplify_changed

    return term1


def terms_equal(term1: Term, term2: Term, *terms) -> bool:
    """ Takes minimal of 2 terms, returns True if all the terms have equal labels and equal children.
        Children are compared in order."""
    labels_equal = term1.label == term2.label and all(term1.label == term.label for term in terms)
    if labels_equal:
        lengths_equal = len(term1.children) == len(term2.children) and all(len(term1.children) == len(term.children) for term in terms)
        if lengths_equal:
            children_equal = all(terms_equal(child1, child2, *childs) for child1, child2, *childs in
                                 zip(term1.children, term2.children, *(map(lambda x: x.children, terms))))
            if children_equal:
                return True
    return False


def preprocess(term1: Term) -> Term:
    """ Change  1. a/b to a*b^(-1)
                2. substraction to addition with *(-1) """
    if term1.label == SUB:
        return Term(ADD,
                    [preprocess(term1.children[0])] + [Term(MUL, [Term(-1), preprocess(subtrahend)]) for subtrahend in term1.children[1:]])
    elif term1.label == DIV:
        a, b = term1.children
        return Term(MUL, [preprocess(a), Term(POW, [preprocess(b), Term(-1)])])
    else:
        return Term(term1.label, [preprocess(child) for child in term1.children])


def postprocess(term1: Term) -> Term:
    """ Change back 1. Multiplication by inverse back to division.
                    2. addition with negative coefficients back to subtraction. """
    if term1.label == ADD:
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
        if len(new_summands) == 1:  # this happens when the whole sum is changed to subtraction
            ret = new_summands[0]
            return Term(ret.label, [postprocess(child) for child in ret.children])
        else:
            return Term(ADD, [postprocess(summand) for summand in new_summands])

    elif term1.label == MUL:
        # turns back 4*x^(-1) to 4/x
        will_be_multiplied = []
        will_divide = []
        for child in term1.children:
            # here we omit cases where exponent is for example -log(x) - so x^(-log(x)) is NOT changed to 1/x^log(x), this would be unnecessary.
            # we only care for negative numbers in exponent - change 4*x^(-2) to 4/x^2.
            if child.label == POW and isinstance(child.children[1].label, numbers.Number) and child.children[1].label < 0:
                child.children[1].label = abs(child.children[1].label)  # modify the term
                if child.children[1].label == 1:
                    will_divide.append(child.children[0])
                else:
                    will_divide.append(child)
            else:
                will_be_multiplied.append(child)

        if not will_be_multiplied:
            numerator = Term(1)
        elif len(will_be_multiplied) == 1:
            numerator = will_be_multiplied[0]
        else:
            numerator = Term(MUL, will_be_multiplied)

        if not will_divide:
            return Term(term1.label, [postprocess(child) for child in term1.children])
        elif len(will_divide) == 1:
            return Term(DIV, [postprocess(numerator), postprocess(will_divide[0])])
        else:
            return Term(DIV, [postprocess(numerator), postprocess(Term(MUL, will_divide))])

    # !!! very precise rule: catches only a^-1
    # matter of preference: I am fine with (log(x))^-5 staying that way but i want x^-1 changed to 1/x
    elif term1.label == POW and isinstance(term1.children[1].label, numbers.Number) and term1.children[1].label == -1:
        return Term(DIV, [Term(1), postprocess(term1.children[0])])

    else:
        return Term(term1.label, [postprocess(child) for child in term1.children])


""" Used to mask a change in implementation.
    First simplifications took only term to simplify.
    But one simplification turned out to need also rules_for_simplification dictionary as an argument."""
def add_unused_parameter(f: callable):
    """ decorate a function of 1 argument to get function of 2 arguments that ignores the second one. """
    def foo(term, rules):
        return f(term)

    return foo


@add_unused_parameter
def associativity(term1: Term) -> (Term, bool):
    """ (* (* x 2) 3)  -->  (* x 2 3)"""
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
    """ (+ 1)  -->  1 """
    assert term1.label in [ADD, MUL]
    if len(term1.children) == 1:
        return term1.children[0], True
    return term1, False


@add_unused_parameter
def distributive_property(term1: Term) -> (Term, bool):
    """ x*x*(1+3)  -->  x*x*1 + x*x*3 """
    for i, child in enumerate(term1.children):
        # look for sum, if found use distributive property
        if child.label == ADD:
            term1.children.pop(i)  # remove that sum, whats left are the terms from product to be distributed
            return Term(ADD, [Term(MUL, children=[summand] + [Term.copy(child) for child in term1.children]) for summand in child.children]), True
    else:  # not found a sum among children
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


@add_unused_parameter
def add_numbers(term1: Term) -> (Term, bool):
    """ 4+3+x+x  -->  7+x+x """
    assert term1.label == ADD

    return perform_calculations_on_numbers_add_mul(term1, operator.add)


@add_unused_parameter
def multiply_numbers(term1: Term) -> (Term, bool):
    """ 4*3*x*x*5 -> 60*x*x """
    assert term1.label == MUL

    return perform_calculations_on_numbers_add_mul(term1, operator.mul)


def perform_calculations_on_numbers_sub_div_pow(term1: Term, operator) -> (Term, bool):
    """ generic function for subtract_numbers, divide_numbers, raise_to_power_numbers.
        it simplifies a term performing term operation on numbers among its children.
        it is seperate to perform_calculations_on_numbers_1 as those operations
        expect binary term. """  # -- only raise_to_power_numbers turned out to be used ;/
    if len(term1.children) == 2:
        left, right = term1.children
        if isinstance(left.label, numbers.Number) and isinstance(right.label, numbers.Number):
            return Term(operator(left.label, right.label)), True
    return term1, False


@add_unused_parameter
def raise_to_power_numbers(term1: Term) -> (Term, bool):
    assert term1.label == POW
    assert len(term1.children) == 2

    return perform_calculations_on_numbers_sub_div_pow(term1, operator.pow)


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
    """ 0+a+b+c  -->  a+b+c """
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
    """ 1*a*b  -->  a*b """
    assert term1.label == MUL
    changed = False
    children = []
    for child in term1.children:
        if child.label == 1:  # then skip it
            changed = True
        else:
            children.append(child)
    #  here we check whether sum becomes empty after this operation
    if not children:  # empty product
        return Term(1), True
    else:
        return Term(MUL, children), changed


@add_unused_parameter
def zero_in_product(term1: Term) -> (Term, bool):
    """ 0*a*b  -->  0 """
    assert term1.label == MUL
    if any(0 == child.label for child in term1.children):
        return Term(0), True
    else:
        return term1, False


# more complex simplification
class ExtractTerm:

    def __init__(self, term, rules):
        self.rules = rules
        self.term = term
        self._children_deepcopy = None

    def generate_index_pairs(self):
        """ This function yields lists to be used in extract_term() and try_to_simplify().

            Here term is a sum and we want to check if there is term that we can extract from the sum.
            For it we need to generate indexes of terms in this sum to be later checked for equality.
            Assume addends in the sum are products. Function for each pair of addends and for each pair of factors in these addends
            yields pair of pairs of indices [(a1, f1), (a2, f2)], where a1, a2 are indices of addends in term1.children
            and f1, f2 are indices of factors in addends.children respectively.
            If addend is not a product but a single term f1 (or f2) is None.

            For example term1 is 4*x+5*x+x and list is [(1, 0), (2, None)]
            this chooses first element in second product (5 from 5*x) and the only element from last 'product' (x from x).
        """

        lengths = []  # will keep number or terms in a product or None if not a product for each summand in term1
        for child in self.term.children:
            if child.label == MUL:
                lengths.append(len(child.children))
            else:
                lengths.append(None)

        for i, annie in enumerate(self.term.children):
            for j, frank in enumerate(self.term.children):
                if i == j:
                    continue

                annie_pairs = []
                if lengths[i] is None:
                    annie_pairs.append((i, None))
                else:
                    for annie_k in range(lengths[i]):
                        annie_pairs.append((i, annie_k))

                for from_annie in annie_pairs:
                    if lengths[j] is None:
                        yield [from_annie, (j, None)]
                    else:
                        for frank_k in range(lengths[j]):
                            yield [from_annie, (j, frank_k)]

    def get_children_copy(self):
        """ try_to_simplify acts on a copy of children, because usually it needs to reverse the changes. """
        return [Term.copy(child) for child in self.term.children]

    def try_to_simplify(self, index_pairs):
        """ take a*b+a*c+d, and indexes [(0, 0), (1, 0)] - indexes choosing both a's,
            and try to simplify (b+c), if simplification reduced (b+c) to single value (d) then add a*(d) to self.term.children instead of a*(b+c).

            Function to be used inside extract_term.
            It takes a term and information on which terms are equal so can be extracted out of the sum.
            It does exactly that - extract the term out of the sum
            and tries to simplify() whats left. If it can simplify it does that and function
            returns extracted_term * whats_simplified among other summands of term1 that are left untouched.
            Further simpification back to the sum is left to distributive_property().
            If no simplification can be made term is untouched - function tries to simplify on copy of terms.

            index_pairs describe where are the equal terms - its a single 2 element list yield by generate_index_pairs.
            List format is a list of pairs (i, j or None) so pairs of 2 indexes or 1 index and None.
            First index corresponds to index in term1.children, second to index in child.children. 'None' means its child itself.
        """
        # need to make a copy of term1.children
        # want to try simplifying, but reverse changes if can't simplify
        children_copy = self.get_children_copy()

        # take any term of those equal terms
        if index_pairs[0][1] is None:  # term is a simple term (i.e: x)
            extracted_term = children_copy[index_pairs[0][0]]
        else:  # term is MUL term (i.e: (* x 4))
            extracted_term = (children_copy[index_pairs[0][0]]).children[index_pairs[0][1]]

        # here its unnecessarily general, due to change of implementation
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
        simplified_term = do_simplifications(term_after_extracted, self.rules)
        simplified = len(simplified_term.children) < old_length or simplified_term.label != ADD  # did it do anything?

        if simplified:
            new_term = Term(MUL, [extracted_term, simplified_term])
            # ^ we rely on distributive property implemented in other function
            # to distribute extracted term back to terms.
            children_left_untouched.append(new_term)  # already touched
            return Term(ADD, children_left_untouched), True
        else:
            return None, False

    def extract_this_term(self) -> (Term, bool):
        """ Simplification:  x+sin(x)+4*sin(x)  -->  x+5*sin(x) .
            Uses generate_index_pairs to get indexes of factors in the sum.
            Checks if factors are equal - if so tries to simplify term using try_to_simplify() by extracting this common factor of the sum. """
        assert self.term.label == ADD
        for index_pairs in self.generate_index_pairs():
            terms_on_indexes = map(lambda pair: self.term.children[pair[0]] if pair[1] is None else self.term.children[pair[0]].children[pair[1]],
                                   index_pairs)
            if terms_equal(*terms_on_indexes):  # here generate_index_pairs only yields index lists of length 2 or more
                new_term, simplified = self.try_to_simplify(index_pairs)
                if simplified:
                    return new_term, simplified
        return self.term, False

    @classmethod
    def extract_term(cls, term, rules) -> (Term, bool):
        """ Function to be put to rules dict. Wrapper around extract_this_term. """
        extract = cls(term, rules)
        return extract.extract_this_term()


@add_unused_parameter
def add_exponents(term1: Term) -> (Term, bool):
    """ a * b ^ c * b --> a * b ^ (c + 1) """
    assert term1.label == MUL
    for i, child_annie in enumerate(term1.children):
        for j, child_frank in enumerate(term1.children):
            if j == i:
                continue
            if child_annie.label == POW:  # annie = a^b
                base_annie, exponent_annie = child_annie.children
                if child_frank.label == POW:  # frank = c^d
                    base_frank, exponent_frank = child_frank.children
                    if terms_equal(base_annie, base_frank):  # a=c; return a^(b+d)
                        # remove ith and jth, append a^(b+d)
                        return Term(term1.label, [term1.children[k] for k in range(len(term1.children)) if k not in [i, j]]
                                                 + [Term(POW, [base_annie, Term(ADD, [exponent_annie, exponent_frank])])]), True
                else:  # frank = f
                    if terms_equal(child_frank, base_annie):  # f=a; a^(b+1)
                        # remove ith and jth, append a^(b+1)
                        return Term(term1.label, [term1.children[k] for k in range(len(term1.children)) if k not in [i, j]]
                                                 + [Term(POW, [base_annie, Term(ADD, [exponent_annie, Term(1)])])]), True

            if terms_equal(child_annie, child_frank):  # a*a=a^2
                return Term(term1.label, [term1.children[k] for k in range(len(term1.children)) if k not in [i, j]]
                                         + [Term(POW, [child_annie, Term(2)])]), True

    return term1, False


@add_unused_parameter
def multiply_exponents(term1: Term) -> (Term, bool):
    assert term1.label == POW
    """ (a^b)^c  --> a^(b*c) """

    if term1.children[0].label == POW:
        return Term(POW, [term1.children[0].children[0], Term(MUL, [term1.children[1], term1.children[0].children[1]])]), True
    return term1, False


""" Rules are applied starting from the last one in the list."""
rules_for_simplification = {ADD: [ExtractTerm.extract_term, associativity, single_term, zero_in_sum, add_numbers],
                            MUL: [associativity, distributive_property, add_exponents, one_in_product, single_term, multiply_numbers, zero_in_product],
                            POW: [one_zero_in_exponent, raise_to_power_numbers, multiply_exponents]}
