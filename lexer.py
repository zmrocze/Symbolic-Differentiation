from symbols_table import *


class InvalidCharacter(Exception):
    pass


def tokenize(text : str) -> list:
    tokens = []
    ind = 0
    while ind < len(text):
        # TODO parse negative numbers
        if text[ind].isdigit():  # digit
            ind_joe = ind
            # will catch fragment of digits and commas,
            # strings like 1.4.56 will throw an error later in the try block
            while ind_joe < len(text) and (text[ind_joe].isdigit() or text[ind_joe] == '.'):
                ind_joe += 1

            try:
                num = int(text[ind:ind_joe])
            except ValueError:
                try:
                    num = float(text[ind:ind_joe])
                except ValueError:
                    raise InvalidCharacter("Can't convert to float or int.")
            tokens.append(num)
            ind = ind_joe

        else:  # letter
            ind_joe = ind
            while ind_joe < len(text) and text[ind:ind_joe] not in symbols:
                ind_joe += 1
            if text[ind:ind_joe] not in symbols:
                raise InvalidCharacter("Unknown function symbol.")
            tokens.append(text[ind:ind_joe])
            ind = ind_joe

    return tokens


if __name__ == "__main__":

    text = "log(x^x*x+cos((x*x^x+x*log(x/x)))/(sin((x*x+x^(x+x))^(x+x*(x+x*x)+x*x+x))^(sin(x)+x)))^x/x*x"
    tokens = tokenize(text)
    print(tokens)
