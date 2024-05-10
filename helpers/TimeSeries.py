import logging
import math

import numpy as np
from nltk.probability import FreqDist
from typing import List
import random

logger = logging.getLogger(__name__)


def construct_fts_from_tokens(token_words: List[str]) -> List[int]:
    """
    :param token_words: List of tokens
    :return: List of integers, each representing frequency of a word at given index
    """
    logger.info("construct_fts_from_tokens called for token_words: {}", token_words)
    freq = FreqDist(token_words)
    freq_dict = dict(freq.most_common())
    fts = [freq_dict[element] for element in token_words]
    logger.debug("construct_fts result: {}", fts)
    return fts


def construct_lts_from_tokens(token_words: List[str]) -> List[int]:
    """
    :param token_words: List of tokens
    :return: List of integers, each representing the length of a word at given index
    """
    logger.info("construct_lts_from_tokens called for token_words: {}", token_words)
    lts = [len(element) for element in token_words]
    logger.debug("construct_lts result: {}", lts)
    return lts


def construct_its(time_series: List[int]) -> List[float]:
    """
    :param time_series:
    :return:
    """
    mean_value = np.mean(time_series)
    integrated_time_series = np.cumsum(time_series - mean_value)
    return integrated_time_series


def construct_its_from_normalized_ts(time_series: List[float]) -> List[float]:
    """
    :param time_series:
    :return:
    """
    integrated_time_series = np.cumsum(time_series)
    return integrated_time_series


def normalize(time_series: List[int]) -> List[float]:
    """
    :param time_series:
    :return:
    """
    mean_value = np.mean(time_series)
    sigma = np.sqrt(np.mean(np.dot(time_series, time_series) - mean_value * mean_value))
    normalized_time_series = (time_series - mean_value) / sigma
    return normalized_time_series


def shuffle(time_series: List[float]) -> List[float]:
    """
    :param time_series:
    :return:
    """
    series_length = len(time_series)
    indices = list(range(series_length))
    random.shuffle(indices)
    time_series_shuffled = [time_series[i] for i in indices]
    return time_series_shuffled


def calculate_cumulative_means(time_series: List[int]) -> {List[int], List[float]}:
    """
    :param time_series:
    :return:
    """
    cumulative_means = []
    i = []
    sum_so_far = 0
    for ii, num in enumerate(time_series, 1):
        sum_so_far += num
        mean_so_far = sum_so_far / ii
        cumulative_means.append(mean_so_far)
        i.append(ii)
    return i, cumulative_means


def calculate_cumulative_variances(time_series: List[int]) -> {List[int], List[float]}:
    """
    :param time_series:
    :return:
    """
    n = len(time_series)
    cumulative_variances = []
    i = []
    sum_so_far = 0
    sum_squared_so_far = 0

    for ii, num in enumerate(time_series, 1):
        sum_so_far += num
        sum_squared_so_far += num ** 2

        mean_so_far = sum_so_far / ii
        sum_squared_so_far += (num - mean_so_far) ** 2
        variance_so_far = sum_squared_so_far / ii

        cumulative_variances.append(variance_so_far)
        i.append(ii)

    return i, cumulative_variances


def calculate_cumulative_skewness(time_series: List[int]) -> {List[int], List[float]}:
    """
    :param time_series:
    :return:
    """
    n = len(time_series)
    cumulative_skewness = []
    i = []
    sum_so_far = 0
    sum_squared_so_far = 0
    sum_cubed_so_far = 0

    for ii, num in enumerate(time_series, 1):
        sum_so_far += num
        mean_so_far = sum_so_far / ii
        sum_squared_so_far += (num - mean_so_far) ** 2
        sum_cubed_so_far += (num - mean_so_far) ** 3

        if ii < 3:  # Handling for cases with fewer than 3 elements
            skewness_so_far = 0
        else:
            skewness_so_far = (sum_cubed_so_far / ii) / math.pow(sum_squared_so_far / ii, 3 / 2)

        cumulative_skewness.append(skewness_so_far)
        i.append(ii)

    return i, cumulative_skewness


def calculate_cumulative_kurtosis(time_series: List[int]) -> {List[int], List[float]}:
    """
    :param time_series:
    :return:
    """
    n = len(time_series)
    cumulative_kurtosis = []
    i = []

    sum_so_far = 0
    sum_squared_so_far = 0
    sum_quartic_so_far = 0

    for ii, num in enumerate(time_series, 1):
        sum_so_far += num
        mean_so_far = sum_so_far / ii
        sum_squared_so_far += (num - mean_so_far) ** 2
        sum_quartic_so_far += (num - mean_so_far) ** 4

        if ii < 3:  # Handling for cases with fewer than 3 elements
            kurtosis_so_far = 0
        else:
            kurtosis_so_far = (sum_quartic_so_far / ii) / (sum_squared_so_far / ii) ** 2 - 3

        cumulative_kurtosis.append(kurtosis_so_far)
        i.append(ii)

    return i, cumulative_kurtosis
