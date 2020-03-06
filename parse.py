from main import Term
from symbols_table import *
import numbers


class InvalidToken(Exception):
    pass

class InvalidExpression(Exception):
    pass

END = 'END OF TOKEN SEQUENCE'
# check for current_token == END


def no_nulls(term1: Term) -> bool:
    """if an invalid expression is provided 'None's will pop up in the tree"""
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

    def parse_binary_operation(self, operations, parse_left, parse_right) -> Term:
        left = parse_left()
        if self.current_token in operations:
            add_or_sub = self.pop_token()
            right = parse_right()
            return Term(add_or_sub, children=[left, right])
        else:
            return left

    def parse_sum(self) -> Term:
        """S -> T | T+S | T-S."""
        return self.parse_binary_operation([ADD, SUB], self.parse_product, self.parse_sum)

    def parse_product(self) -> Term:
        """T -> F | F*T | F/T. """
        return self.parse_binary_operation([MUL, DIV], self.parse_power, self.parse_product)

    def parse_power(self) -> Term:
        """F -> E | E^E."""
        # need to be explicit with brackets and write x^(x^x) or (x^x)^x
        return self.parse_binary_operation([POW], self.parse_term, self.parse_term)

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
