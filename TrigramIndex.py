import os
import re
import json
from collections import defaultdict

import nltk
from nltk import RegexpTokenizer, word_tokenize, pos_tag
import numpy as np
from bitarray import bitarray


class TrigramIndex:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.index = defaultdict(list)
        self.build_index()


    def build_index(self):
        for term in self.vocabulary:
            term_with_end = '$' + term + '$'
            trigrams = [term_with_end[i:i + 3] for i in range(len(term_with_end) - 2)]
            for trigram in trigrams:
                if trigram not in self.index.keys():
                    self.index[trigram] = [term]
                else:
                    self.index[trigram].append(term)

    def search(self, query):
        results = None

        search_query = '$' + query + '$'
        for i in range(len(search_query) - 2):
            trigram = search_query[i:i + 3]
            if '*' in trigram:
                continue
            next_terms = set(self.index.get(trigram, []))
            if results is None:
                results = next_terms
            else:
                results &= next_terms

        # If no results are found, return an empty list
        if not results:
            return []

        # If vocabulary is a dictionary, get the corresponding terms
        return list(results)

    def print_index(self):
        for triram, files in self.index.items():
            print(f"{triram}: {files}")
