import logging
import nltk
from nltk.tokenize import word_tokenize
import re
from typing import List

nltk.download('punkt')

logger = logging.getLogger(__name__)


def clean_special_chars(text: str) -> str:
    """
    :param text: Text as one big string
    :return: The same text without special characters
    """
    logger.info("clean_special_chars called for text: {}", text)
    # cut special characters out
    text_cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    logger.debug("clean_special_chars result: {}", text_cleaned)
    return text_cleaned


def perform_text_preprocessing(text: str) -> str:
    """
    :param text: Text on which preprocessing is performed
    :return: Preprocessed text
    """
    # step 1 - to lowercase
    text_lower = text.lower()

    # step 2 - clean special characters
    text_cleaned = clean_special_chars(text_lower)

    return text_cleaned


def tokenize_text(text: str) -> List[str]:
    """
    :param text: Text as one big string
    :return: List of tokens
    """
    logger.info("tokenize_text called for text: {}", text)
    token_words = word_tokenize(text)
    logger.debug("tokenize_text result: {}", token_words)
    return token_words
