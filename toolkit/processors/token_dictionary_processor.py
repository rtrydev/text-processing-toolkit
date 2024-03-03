from itertools import count
from typing import Dict, List, Set
from toolkit.classes.token import Token


class TokenDictionaryProcessor:
    @staticmethod
    def create_dictionary(token_sequence_collection: List[List[Token]]) -> Dict[str, Token]:
        iter_count = count(5)
        result: Dict[str, Token] = {}
        entries: Set[str] = set()

        for sequence in token_sequence_collection:
            for token in sequence:
                if token.body in entries:
                    continue

                entries.add(token.body)
                if token.data.get('id') is None:
                    token.data['id'] = next(iter_count)

                result[token.body] = token

        return result

    @staticmethod
    def process_token_sequence_to_numerical(tokens: List[Token], token_dictionary: Dict[str, Token]) -> List[int]:
        return [
            token_dictionary[token.body].data['id']
            for token in tokens
        ]
