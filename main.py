from simplification import simplify
from printing import to_pretty_string
from derivatives import derivative, rules_for_differentiation
from lexer import tokenize, ParsingException
from parse import Parser


def main_loop():
    formula = input("enter function to differentiate: ")
    try:
        tokens = tokenize(formula)
        parser = Parser(tokens)
        term1 = parser.parse()
        # print(f"term tree: {term1}")
    except ParsingException as e:
        print(e)
    else:
        term1_derivative = derivative(term1, rules_for_differentiation)
        # print(f"derivative tree: {deriv_term1}")
        # print(to_pretty_string(deriv_term1))
        simplified_derivative = simplify(term1_derivative)
        pretty_derivative = to_pretty_string(simplified_derivative)
        print(f"derivative: {pretty_derivative}\n")


if __name__ == "__main__":
    while True:
        main_loop()
