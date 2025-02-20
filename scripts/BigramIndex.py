import os
import re
import json
from collections import defaultdict

import nltk
from nltk import RegexpTokenizer, word_tokenize, pos_tag
import numpy as np
from bitarray import bitarray


class BigramIndex:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.filenames = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
        self.index = defaultdict(list)
        self.build_index()

    def map(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read().lower().replace("\n", " ")
        tagged_words = word_tokenize(text)
        bigrams = [(tagged_words[i], tagged_words[i + 1]) for i in range(len(tagged_words) - 1)]
        return bigrams

    def build_index(self):
        for filename in self.filenames:
            bigrams = self.map(filename)
            doc_name = os.path.basename(filename)
            for bigram in bigrams:
                if doc_name not in self.index[bigram]:
                    self.index[bigram].append(doc_name)

    def phrase_search(self, query):
        words = query.lower().split()
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        result = set(self.index[bigrams[0]])
        for bigram in bigrams[1:]:
            result &= set(self.index[bigram])
        return sorted(list(result))

    def print_index(self):
        for bigram, files in self.index.items():
            print(f"{bigram}: {files}")
