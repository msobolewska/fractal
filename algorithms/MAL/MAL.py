from typing import List

import numpy as np
import statistics

from scipy.optimize import curve_fit

from helpers.TextManipulations import tokenize_text_merge_meaningless, split_into_syllables, split_into_phonemes

syllables_length_in_word_dict = {}


def create_sentences_length_in_paragraph_dict(sentences: List[str]) -> {}:
    """
    :param sentences:
    :return:
    """
    sentences_length_in_paragraph_dict = {}
    for sublist in sentences:
        length = len(sublist)
        if length not in sentences_length_in_paragraph_dict:
            sentences_length_in_paragraph_dict[length] = []
        sentences_length_in_paragraph_dict[length].append(sublist)
    return sentences_length_in_paragraph_dict


def perform_word_split(sentences_dict: {}) -> {}:
    """
    :param sentences_dict:
    :return:
    """
    sentences_dict_split_words = \
        {key: [[tokenize_text_merge_meaningless(elem) for elem in sublist] for sublist in value] for key, value in
         sentences_dict.items()}

    return sentences_dict_split_words


def perform_syllable_split(words_dict: {}) -> {}:
    """
    :param words_dict:
    :return:
    """
    words_dict_split_syllables = \
        {key: [[split_into_syllables(elem) for elem in sublist] for sublist in value] for key, value in
         words_dict.items()}

    return words_dict_split_syllables


def perform_phoneme_split(syllables_dict: {}) -> {}:
    """
    :param syllables_dict:
    :return:
    """
    syllables_dict_split_syllables = \
        {key: [[split_into_phonemes(elem) for elem in sublist] for sublist in value] for key, value in
         syllables_dict.items()}

    return syllables_dict_split_syllables


def create_words_length_in_sentence_dict(sentences_dict_split: {}) -> {}: # TODO rename to something more general
    """
    :param sentences_dict_split:
    :return:
    """
    words_length_in_sentence_dict = {}
    for key, value in sentences_dict_split.items():
        for sublist in value:
            for subsublist in sublist:
                length = len(subsublist)
                if length not in words_length_in_sentence_dict:
                    words_length_in_sentence_dict[length] = []
                words_length_in_sentence_dict[length].append(subsublist)

    return words_length_in_sentence_dict


def count_sentences_length_in_paragraph_dict(sentences_dict_split: {}) -> {}: # TODO rename to somtheing more general
    """
    :param sentences_dict_split:
    :return:
    """

    sentences_dict_split_counted = \
        {key: [[len(subsublist) for subsublist in sublist] for sublist in value] for key, value in
         sentences_dict_split.items()}

    return sentences_dict_split_counted


def average_sentences_length_in_paragraph_dict_single(sentences_dict_split_counted: {}) -> {}:
    """
    :param sentences_dict_split_counted:
    :return:
    """
    sentences_dict_split_average_single = \
        {key: [statistics.mean(sublist) for sublist in value] for key, value in
         sentences_dict_split_counted.items()}

    return sentences_dict_split_average_single


def average_sentences_length_in_paragraph_dict(sentences_dict_split_counted_average_single: {}) -> {}:
    """
    :param sentences_dict_split_counted_average_single:
    :return:
    """
    sentences_dict_split_average = \
        {key: statistics.mean(value) for key, value in
         sentences_dict_split_counted_average_single.items()}

    return sentences_dict_split_average


def print_sorted(dict_in: {}) -> None:
    """
    :param dict_in:
    :return:
    """
    sorted_dict = dict(sorted(dict_in.items()))

    print("Key\tValue")
    print("-" * 15)
    for key, value in sorted_dict.items():
        print(f"{key}\t{value}")


def model_func(x, a, b, c):
    """
    :param x:
    :param a:
    :param b:
    :param c:
    :return:
    """
    return a * pow(x, -b) * np.exp(-c * np.log(x))


def model_func_truncated(x, a, b):
    """
    :param x:
    :param a:
    :param b:
    :return:
    """
    return a * pow(x, -b)


def find_A_b_numerically_truncated(X_Y: {}) -> {float, float}:
    """
    :param X_Y:
    :return:
    """
    x_data = list(X_Y.keys())
    y_data = list(X_Y.values())
    popt, pcov = curve_fit(model_func_truncated, x_data, y_data, bounds=([1e-6, 1e-6], [np.inf, np.inf]))
    a_fit, b_fit = popt
    return a_fit, b_fit


def find_A_b_numerically_full(X_Y: {}) -> {float, float, float}:
    """
    :param X_Y:
    :return:
    """
    x_data = list(X_Y.keys())
    y_data = list(X_Y.values())
    popt, pcov = curve_fit(model_func, x_data, y_data, bounds=([1e-6, 1e-6, 1e-6], [np.inf, np.inf, np.inf]))
    a_fit, b_fit, c_fit = popt
    return a_fit, b_fit, c_fit
