import os
import re
import json
import nltk
from nltk import RegexpTokenizer, word_tokenize
import numpy as np
from bitarray import bitarray

operators = ['and', 'or', 'not']

class IncidenceMatrix:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.filenames = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
        self.term_to_bitarray = {}
        self.build_matrix()

    def map(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read().lower().replace("\n", " ")
        words = word_tokenize(text)
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        return word_counts

    def build_matrix(self):
        documents_word_counts = []

        # Collect word counts for each document
        for filename in self.filenames:
            word_counts = self.map(filename)
            documents_word_counts.append(word_counts)

        # Initialize bitarrays for each term
        for doc_idx, word_counts in enumerate(documents_word_counts):
            for word in word_counts.keys():
                if word not in self.term_to_bitarray:
                    self.term_to_bitarray[word] = bitarray(len(self.filenames))
                    self.term_to_bitarray[word].setall(0)
                self.term_to_bitarray[word][doc_idx] = 1
        #print(self.term_to_bitarray)

    def tokenize_query(self, query):
        tokens = word_tokenize(query.lower())
        pairs = [['or', tokens[0]]]
        count = 1
        for token in tokens[1:]:
            if token in operators:
                if token == 'not':
                    pairs[count][0] += f" {token}"
                else:
                    pairs.append([token])
            else:
                if len(pairs[-1]) == 1:
                    pairs[-1].append(token)
                else:
                    pairs.append(['or', token])

        return pairs

    def boolean_search(self, query):
        query_pairs = self.tokenize_query(query)

        files = set(range(len(self.filenames)))

        result = None

        # Perform boolean operations on the result set
        for op, term in query_pairs:
            if term in self.term_to_bitarray:
                term_bitarray = self.term_to_bitarray[term]
                term_docs = {i for i, bit in enumerate(term_bitarray) if bit}
            else:
                term_docs = set()

            if result is None:
                result = term_docs


            if op == 'and':
                result &= term_docs
            elif op == 'or':
                result |= term_docs
            elif op == 'and not':
                result &= files - term_docs
            elif op == 'or not':
                result |= files - term_docs

        # Convert result set indices to filenames
        return [self.filenames[idx].replace("docs\\", "") for idx in result]

    def print_matrix(self):
        print("Матриця інцидентности (термін-документ):")
        print("Терміни/Документи", end="\t")
        for filename in self.filenames:
            print(os.path.basename(filename), end="\t")
        print()

        for term, bitarr in self.term_to_bitarray.items():
            print(term, end="\t")
            for bit in bitarr:
                print(int(bit), end="\t")
            print()

    def get_size(self):
        print(len(self.term_to_bitarray))

