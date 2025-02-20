import os
from builtins import enumerate

from nltk import word_tokenize
from suffix_tree import Tree
from suffix_tree.node import Internal, Leaf



class SuffixTree:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.tree = Tree({i: " " + w + " " for i, w in enumerate(self.vocabulary)})

    def search(self, query):
        tokens = query.split("*")
        tokens[0] = " " + tokens[0]
        tokens[-1] += " "

        results = None
        for tok in tokens:
            matches = {k for k, path in self.tree.find_all(tok) if path}
            if not matches:
                return []
            if results is None:
                results = matches
            else:
                results &= matches
        return [self.vocabulary[result] for result in results]


    def print_index(self):
        def visitor(node, depth=0):
            indent = ' ' * depth * 2
            if isinstance(node, Internal):
                print(f"{indent}Internal '{node}'")
                for child in node.children.values():
                    visitor(child, depth + 1)
            elif isinstance(node, Leaf):
                print(f"{indent}Leaf '{node}'")
            else:
                raise ValueError(f"Unknown node type: {type(node)}")

        print("Suffix Tree:")
        self.tree.pre_order(visitor)
