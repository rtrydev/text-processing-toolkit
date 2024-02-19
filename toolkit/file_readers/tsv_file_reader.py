from typing import List

from toolkit.classes.token import Token
from toolkit.enums.language import Language
from toolkit.factories.tokenizer_factory import TokenizerFactory
from toolkit.file_readers.base_file_reader import BaseFileReader
from toolkit.processors.token_sequence_enhancer import TokenSequenceEnhancer


class TsvFileReader(BaseFileReader):
    def tokenize_file(self, file_path: str, line_limit: int = -1) -> List[List[Token]]:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        segment_collections = [
            self.__get_segments(line)
            for line in (lines[:line_limit] if line_limit != -1 else lines)
        ]

        ja_tokenizer = TokenizerFactory.create_tokenizer(Language.JAPANESE)
        en_tokenizer = TokenizerFactory.create_tokenizer(Language.ENGLISH)

        enhancer = TokenSequenceEnhancer()

        tokenized_sequences = [
            enhancer.join_separated_sequences(
                ja_tokenizer.tokenize_string(segments[0]),
                en_tokenizer.tokenize_string(segments[1])
            ) for segments in segment_collections
        ]
        max_len = max(map(
            lambda segment: len(segment),
            tokenized_sequences
        ))

        return [
            enhancer.pad_sequence(sequence, max_len + 2)
            for sequence in tokenized_sequences
        ]

    def __get_segments(self, line: str):
        splits = line.split('\t')
        return [
            splits[1],
            splits[3]
        ]
