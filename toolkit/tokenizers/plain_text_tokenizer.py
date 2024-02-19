from typing import List

from nltk.tokenize import word_tokenize

from toolkit.classes.token import Token
from toolkit.tokenizers.base_tokenizer import BaseTokenizer


class PlainTextTokenizer(BaseTokenizer):
    def tokenize_string(self, line: str) -> List[Token]:
        raw_tokens = word_tokenize(line)

        return [
            Token(
                body=raw_token,
                data={}
            ) for raw_token in raw_tokens
        ]
