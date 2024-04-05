import logging
import nltk
from nltk.tokenize import word_tokenize
import re
from typing import List
import pronouncing
import pyphen


nltk.download('punkt')

dic = pyphen.Pyphen(lang='en')

logger = logging.getLogger(__name__)

meaningless_words = {"a", "about", "above", "across", "after", "against", "along", "amid", "among", "around", "as",
                     "at", "before", "behind", "below", "beneath", "beside", "besides", "between", "beyond", "but",
                     "by", "concerning", "considering", "despite", "down", "during", "except", "for", "from",
                     "in", "inside", "into", "like", "near", "next", "of", "off", "on", "onto", "out",
                     "outside", "over", "past", "regarding", "round", "since", "through", "throughout", "till", "to",
                     "toward", "towards", "the", "under", "underneath", "until", "unto", "up", "upon", "with", "within",
                     "without"}


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


def tokenize_text_merge_meaningless(text: str) -> List[str]:
    """
    :param text:
    :return:
    """
    tokens = word_tokenize(text)
    merged_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i].lower() in meaningless_words:
            merged_token = tokens[i]
            i += 1
            while i < len(tokens) and tokens[i].lower() in meaningless_words:
                merged_token = merged_token + " " + tokens[i]
                i += 1
            if i < len(tokens):
                merged_token = merged_token + " " + tokens[i]
                i += 1
            merged_tokens.append(merged_token)
        else:
            merged_tokens.append(tokens[i])
            i += 1
    return merged_tokens


def split_text_into_paragraphs(text: str) -> List[str]:
    """
    :param text:
    :return:
    """
    text_split = text.split('\n\n')
    text_split_nonnull = list(filter(lambda s: s.strip() != "", (s.replace('\n', ' ') for s in text_split)))
    return text_split_nonnull


def split_text_into_sentences(text: str) -> List[str]:
    """
    :param text:
    :return:
    """
    sentences = nltk.sent_tokenize(text)
    return sentences


def split_into_syllables(text: str) -> List[str]:
    """
    :param text:
    :return:
    """
    words = text.split()
    syllables_list = [dic.inserted(word).split('-') for word in words]
    merged_syllables = sum(syllables_list, [])
    return merged_syllables


def split_into_phonemes(text: str) -> List[str]:
    """
    :param text:
    :return:
    """
    words = text.split()
    phonemes_list = []

    # Process each word separately
    for word in words:
        # If the word is "a", add it directly to the result list
        if word == "a":
            phonemes_list.append(["AH0"])
        else:
            # Get phonemes for the word
            phonemes = pronouncing.phones_for_word(word)
            # Split the phonemes into individual phonemes and append to the result list
            if phonemes:
                for phoneme in phonemes:
                    phonemes_list.append(phoneme.split())

    # Merge the phonemes into a single list
    merged_phonemes = sum(phonemes_list, [])
    return merged_phonemes
