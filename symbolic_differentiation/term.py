from functools import reduce


class Term:

    def __init__(self, label: str or int, children=None):
        self.label = label
        self.children = children if children else []

    def __repr__(self):
        if self.children:
            return f"({self.label} " + reduce(lambda str1, str2: str1+" "+str2, (repr(child) for child in self.children)) + ")"
        else:
            return f"{self.label}"

    @classmethod
    def copy(cls, obj):
        """makes a copy of obj"""
        return cls(obj.label, [cls.copy(child) for child in obj.children])
