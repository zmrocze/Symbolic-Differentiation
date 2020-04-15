from simplification import do_simplifications, rules_for_simplification
from printing import to_pretty_string
from derivatives import derivative, rules_for_differentiation
from lexer import tokenize
from parse import Parser


formula = "8*x*5*x*2*3"
tokens = tokenize(formula)
parser = Parser(tokens)
term1 = parser.parse()
deriv_term1 = derivative(term1, rules_for_differentiation)
simplified_deriv = do_simplifications(deriv_term1, rules_for_simplification)
print(to_pretty_string(simplified_deriv))