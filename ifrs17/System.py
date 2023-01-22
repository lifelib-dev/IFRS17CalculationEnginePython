from typing import Generic



class Grouping(list):

    def __init__(self, key, values):
        self.Key = key
        super().__init__(values)


# def SelectMany(nested: list[list], selector1, selector2):
#     for l in nested:


