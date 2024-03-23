import logging

logger = logging.getLogger(__name__)

TEXTS_LOC = "texts/"  # location of catalogue ith text files relative to notebooks


def read_txt_file(filename: str) -> str:
    logger.info("read_txt_file called with {}", filename)
    """
    :param filename: Name of a txt file to import, without the extension
    :return:
    """
    with open(TEXTS_LOC + filename + '.txt', 'r') as file:
        data = file.read().replace('\n', ' ')
        return data
