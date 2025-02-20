import os
import fnmatch

from nltk import word_tokenize


class PermutermIndex:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.filenames = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
        self.index = {}
        self.build_index()

    def add_term(self, term):
        term_with_end = term + '$'
        rotations = [term_with_end[i:] + term_with_end[:i] for i in range(len(term_with_end))]
        for rotation in rotations:
            self.index[rotation] = term

    def build_index(self):
        for filename in self.filenames:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read().lower().replace("\n", " ")
            terms = word_tokenize(text)
            for term in terms:
                self.add_term(term)

    def search(self, query):
        query_parts = query.split('*')
        if len(query_parts) == 1:
            # No wildcard in the query
            return {query} if query in self.index else set()

        # Multiple wildcards
        initial_part = query_parts[-1] + '$' + query_parts[0]
        matching_terms = [term for term in self.index if term.startswith(initial_part)]
        results = set()

        for term in matching_terms:
            original_term = self.index[term]
            if self.match_with_wildcards(original_term, query_parts):
                results.add(original_term)

        return results

    def match_with_wildcards(self, term, query_parts):
        # Check if the term matches the query with wildcards
        start = 0
        for part in query_parts:
            idx = term.find(part, start)
            if idx == -1:
                return False
            start = idx + len(part)
        return True

    def print_index(self):
        for permuted_term, terms in self.index.items():
            print(f"{permuted_term}: {', '.join(terms)}")