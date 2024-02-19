from typing import List

from fugashi import Tagger # type: ignore

from toolkit.classes.token import Token
from toolkit.tokenizers.base_tokenizer import BaseTokenizer


class JapaneseTextTokenizer(BaseTokenizer):
    def __init__(self):
        self.dict_tagger = Tagger('-Owakati')

    def tokenize_string(self, line: str) -> List[Token]:
        self.dict_tagger.parse(line)

        return [
            Token(
                body=str(tag),
                data={}
            ) for tag in self.dict_tagger(line)
        ]
