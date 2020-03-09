from term import Term
from symbols_table import *
import numbers


class InvalidToken(Exception):
    pass

class InvalidExpression(Exception):
    pass

END = 'END OF TOKEN SEQUENCE'
# check for current_token == END


def no_nulls(term1: Term) -> bool:
    """returns True if there are no None's in term labels.
    if an invalid expression is parsed 'None's will pop up in the tree"""
    if term1 is None:
        return False
    else:
        return all((no_nulls(child) for child in term1.children))


class Parser:

    def __init__(self, tokens):
        self.index = 0
        self.tokens = tokens

    @property
    def current_token(self):
        try:
            token = self.tokens[self.index]
        except IndexError:
            return END  # could really just be None
        return token

    def pop_token(self):
        token = self.current_token
        self.index += 1
        return token

    def parse(self) -> Term:
        """checks if expression was valid, a wrapper around parse_sum"""
        S = self.parse_sum()
        if self.current_token == END and no_nulls(S):  # i believe no_nulls(S) is enough, don't need to check for END
            return S
        else:
            raise InvalidExpression("Provided expression is not legal.")
    #
    # def parse_binary_operation_recursive(self, operations, parse_left, parse_right) -> Term:
    #     """Unused.
    #     Function used to parse expression of type: S -> A | A+S. so a right recursive one.
    #     Changed this when realized it parses - and / wrong
    #     as those are only left associative"""
    #     left = parse_left()
    #     if self.current_token in operations:
    #         add_or_sub = self.pop_token()
    #         right = parse_right()
    #         return Term(add_or_sub, children=[left, right])
    #     else:
    #         return left

    def parse_binary_operation(self, operations: list, parse_component: callable) -> Term:
        """parses expression: A -> B { operation B}"""
        result_term = parse_component()
        while self.current_token in operations:
            result_term = Term(self.pop_token(), children=[result_term, parse_component()])
        return result_term

    def parse_sum(self) -> Term:
        """S -> T { + T | - T}."""
        return self.parse_binary_operation([ADD, SUB], self.parse_product)

    def parse_product(self) -> Term:
        """T -> F { * F | / F}. """
        return self.parse_binary_operation([MUL, DIV], self.parse_power)

    def parse_power(self) -> Term:
        """F -> E { ^ E}."""
        return self.parse_binary_operation([POW], self.parse_term)

    def parse_term(self) -> Term:
        """E -> num | var | (S) | sin(S) | log(S) | cos(S) | tan(S)."""
        if isinstance(self.current_token, numbers.Number):
            return Term(self.pop_token())
        elif self.current_token == VAR:
            return Term(self.pop_token())
        elif self.current_token == OPEN_BR:
            self.pop_token()  # (
            S = self.parse_sum()
            if self.pop_token() != CLOSE_BR:
                raise InvalidToken("Matching closing bracket expected.")
            return S
        elif self.current_token in [SIN, LOG, TAN, COS]:
            fun = self.pop_token()  # FUN
            if self.pop_token() != OPEN_BR:
                raise InvalidToken("Opening bracket after function symbol expected.")
            S = self.parse_sum()
            if self.pop_token() != CLOSE_BR:
                raise InvalidToken("Matching closing bracket expected after function symbol.")
            return Term(fun, children=[S])


if __name__ == "__main__":
    from lexer import tokenize

    text = "(x^5+x*(5)+3.14^4)+5"
    tokens = tokenize(text)
    parser = Parser(tokens)
    term = parser.parse()
    print(term)
