import multiprocessing
from collections import defaultdict
from multiprocessing.pool import Pool
from functools import partial
import os
import re
import json
import nltk

from nltk.tokenize import word_tokenize

from BigramIndex import BigramIndex
from IncidenceMatrix import IncidenceMatrix
from InvertedIndex import InvertedIndex
from PermutermIndex import PermutermIndex
from SuffixTree import SuffixTree
from TrigramIndex import TrigramIndex


def map(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().lower().replace("\n", " ")
    words = word_tokenize(text)
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

def merge_word_counts(word_counts1, word_counts2):
    merged_word_counts = word_counts1.copy()
    for word, count in word_counts2.items():
        merged_word_counts[word] = merged_word_counts.get(word, 0) + count
    return merged_word_counts

#
# Складність алгоритму - O(N) для функції map (N - кількість слів),
# O(F*T) для функції reduce (F - кількість файлів, T - кількість термінів)
#
def build_dict(pool, filenames):
    # Обчислення частоти слів у кожному файлі
    word_counts_list = pool.map(map, filenames)

    # Створення часткової функції для об'єднання словників
    merge_word_counts_partial = partial(merge_word_counts)

    # Об'єднання словників з частотою слів
    global_word_counts = word_counts_list[0]
    for word_counts in word_counts_list[1:]:
        global_word_counts = merge_word_counts_partial(global_word_counts, word_counts)

    # Збереження словника термінів
    with open('dictionary.json', 'w', encoding='utf-8') as f:
        json.dump(global_word_counts, f)

    # Оцінка розміру колекції, загальної кількості слів у колекції та розміру словника
    collection_size = sum(os.path.getsize(filename) for filename in filenames)
    total_word_count = sum(global_word_counts.values())
    dictionary_size = os.path.getsize('dictionary.json')

    print(f"Розмір колекції: {collection_size} байт")
    print(f"Загальна кількість слів у колекції: {total_word_count}")
    print(f"Розмір словника: {dictionary_size} байт")

def print_indices():
    incidence_matrix.print_matrix()
    inverted_index.print_index()
    bigram_index.print_index()

    incidence_matrix.get_size()
    inverted_index.get_size()

def bool_search():
    query = "be and not says"
    results_mat = incidence_matrix.boolean_search(query)
    print("Результати булевого пошуку (матриця інцидентності):")
    print(results_mat)

    results_ind = inverted_index.boolean_search(query)
    print("Результати булевого пошуку (інвертований індекс):")
    print(results_ind)

def phrase_search():
    search_query = "basely /2 Amy /4 state"

    phrase_bi = bigram_index.phrase_search(search_query)
    print("Результати фразового пошуку (двослівний індекс):")
    print(phrase_bi)

    phrase_inv = inverted_index.phrase_search(search_query)
    print("Результати фразового пошуку (інвертований індекс):")
    print(phrase_inv)

    phrase_len = inverted_index.length_search(search_query)
    print("Результати пошуку з урахуванням відстані (інвертований індекс):")
    print(phrase_len)


if __name__ == '__main__':
    # Шлях до папки з текстовими файлами
    data_dir = 'docs'

    with open("dictionary.json", 'r') as f:
        inv_index = json.load(f)

    vocabulary = list(inv_index.keys())

    # Отримання списку файлів у папці
    filenames = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]

    # Створення пулу для паралельної обробки файлів
    pool = Pool(processes=multiprocessing.cpu_count())

    build_dict(pool, filenames)

    incidence_matrix = IncidenceMatrix(data_dir)
    inverted_index = InvertedIndex(data_dir)
    bigram_index = BigramIndex(data_dir)
    suffix_tree = SuffixTree(vocabulary)
    permuterm_index = PermutermIndex(data_dir)
    permuterm_index.print_index()
    suffix_tree.print_index()


    print_indices()
    bool_search()
    phrase_search()
    print(suffix_tree.search("s*op"))
    print(permuterm_index.search("s*op"))

    suffix_tree = SuffixTree(vocabulary)
    print(suffix_tree.search("s*h*p"))

    permuterm_index = PermutermIndex(data_dir)
    print(permuterm_index.search("s*h*p"))

    triram_index = TrigramIndex(vocabulary)
    print(triram_index.search("s*h*p"))

    triram_index.print_index()



