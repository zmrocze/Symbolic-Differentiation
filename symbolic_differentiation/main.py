# from simplification import simplify
# from printing import to_pretty_string
# from derivatives import derivative
# from lexer import ParsingException
# from parse import parse
from simplification import simplify
from printing import to_pretty_string
from derivatives import derivative
from lexer import ParsingException
from parse import parse

def main_loop():
    formula = input("enter function to differentiate: ")
    try:
        term1 = parse(formula)
        # print(f"term tree: {term1}")
    except ParsingException as e:
        print(e)
    else:
        term1_derivative = derivative(term1)
        # print(f"derivative tree: {deriv_term1}")
        # print(to_pretty_string(deriv_term1))
        simplified_derivative = simplify(term1_derivative)
        pretty_derivative = to_pretty_string(simplified_derivative)
        print(f"derivative: {pretty_derivative}\n")


if __name__ == "__main__":
    while True:
        main_loop()
