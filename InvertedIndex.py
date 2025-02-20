import os
import re
from re import match
import json
from collections import defaultdict

import nltk
from nltk import RegexpTokenizer, word_tokenize
import numpy as np
from bitarray import bitarray

operators = ['and', 'or', 'not']

class InvertedIndex:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.filenames = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
        self.index = defaultdict(lambda: {'frequency': 0, 'documents': defaultdict(list)})
        self.filename_to_index = {filename: idx for idx, filename in enumerate(self.filenames)}
        self.build_index()

    def map(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read().lower().replace("\n", " ")
        words = word_tokenize(text)
        word_positions = defaultdict(list)
        for position, word in enumerate(words):
            word_positions[word].append(position)
        return word_positions

    def build_index(self):
        for filename in self.filenames:
            word_positions = self.map(filename)
            doc_name = os.path.basename(filename)
            for word, positions in word_positions.items():
                self.index[word]['frequency'] += len(positions)
                self.index[word]['documents'][doc_name].extend(positions)

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
        query_pair = self.tokenize_query(query)

        files = set(range(len(self.filenames)))

        result = None

        for op, term in query_pair:
            if term in self.index:
                term_docs = {self.filename_to_index[os.path.join(self.data_dir, filename)] for filename in
                             self.index[term]['documents'].keys()}
            else:
                term_docs = set()

            if result is None:
                result = term_docs


            if op == 'and':
                result &= term_docs
            elif op == 'or':
                result |= term_docs
            elif op == 'and not':
                # print(files)
                # print(term_docs)
                result &= files - term_docs
            elif op == 'or not':
                result |= files - term_docs
        return [os.path.basename(self.filenames[idx]) for idx in sorted(result)]

    def phrase_search(self, query):
        query_tokens = query.lower().split()
        phrase_positions = defaultdict(list)
        for position, word in enumerate(query_tokens):
            phrase_positions[word].append(position)

        result = set(self.index[query_tokens[0]]['documents'].keys())
        for word in query_tokens[1:]:
            result &= set(self.index[word]['documents'].keys())

        matching_docs = []
        for doc_name in result:
            first_word_positions = self.index[query_tokens[0]]['documents'][doc_name]
            for starting_position in first_word_positions:
                found = True
                for i in range(1, len(query_tokens)):
                    if starting_position + i not in self.index[query_tokens[i]]['documents'][doc_name]:
                        found = False
                        break
                if found:
                    matching_docs.append(doc_name)
                    break

        return sorted(matching_docs)

    def length_search(self, query):
        search_query = query.lower().split()
        phrase_positions = defaultdict(list)

        query_tokens = search_query[::2]
        distances = list(map(lambda x: int(x[1:]), search_query[1::2]))


        for position, word in enumerate(query_tokens):
            phrase_positions[word].append(position)

        result = set(self.index[query_tokens[0]]['documents'].keys())
        for word in query_tokens[1:]:
            result &= set(self.index[word]['documents'].keys())

        matching_docs = []
        for doc_name in result:
            first_word_positions = self.index[query_tokens[0]]['documents'][doc_name]
            for starting_position in first_word_positions:
                found = True
                current_position = starting_position
                for i in range(1, len(query_tokens)):
                    next_word_positions = self.index[query_tokens[i]]['documents'][doc_name]
                    valid_positions = [pos for pos in next_word_positions if
                                       current_position < pos <= current_position + distances[i - 1]]
                    if not valid_positions:
                        found = False
                        break
                    current_position = valid_positions[0]  # Move to the position of the next word
                if found:
                    matching_docs.append(doc_name)
                    break

        return sorted(matching_docs)

    def print_index(self):
        print("Інвертований індекс:")
        for term, data in self.index.items():
            doc_info = []
            for doc_name, positions in data['documents'].items():
                positions_str = ", ".join(map(str, positions))
                doc_info.append(f"{doc_name}: {positions_str}")
            doc_info_str = "; ".join(doc_info)
            print(f"<{term}: {data['frequency']}; {doc_info_str}>")


    def get_size(self):
        print(len(self.index))