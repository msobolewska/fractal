import logging
import numpy as np
from nltk.probability import FreqDist
from typing import List

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

