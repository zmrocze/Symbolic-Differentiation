from functools import reduce


class Term:

    def __init__(self, label: str or int, children=None):
        # term1.type
        self.label = label
        # term1.type = variable/number/function
        self.children = children if children else []

    def __repr__(self):
        if self.children:
            return f"({self.label} " + reduce(lambda str1, str2: str1+" "+str2, (repr(child) for child in self.children)) + ")"
        else:
            return f"{self.label}"
