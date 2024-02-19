from toolkit.enums.language import Language
from toolkit.tokenizers.base_tokenizer import BaseTokenizer
from toolkit.tokenizers.japanese_text_tokenizer import JapaneseTextTokenizer
from toolkit.tokenizers.plain_text_tokenizer import PlainTextTokenizer


class TokenizerFactory:
    @staticmethod
    def create_tokenizer(language: Language) -> BaseTokenizer:
        if language == Language.ENGLISH:
            return PlainTextTokenizer()

        if language == Language.JAPANESE:
            return JapaneseTextTokenizer()
