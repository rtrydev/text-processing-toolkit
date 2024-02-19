from typing import List
from toolkit.classes.token import Token


class TokenSequenceEnhancer:
    def __init__(self):
        self.START_TOKEN = Token(
            body='<|start|>',
            data={
                'id': 1
            }
        )
        self.END_TOKEN = Token(
            body='<|end|>',
            data={
                'id': 2
            }
        )
        self.PAD_TOKEN = Token(
            body='',
            data={
                'id': 3
            }
        )
        self.SEPARATOR_TOKEN = Token(
            body='<|separator|>',
            data={
                'id': 4
            }
        )

    def pad_sequence(self, sequence: List[Token], desired_len: int = 0) -> List[Token]:
        sequence.insert(0, self.START_TOKEN)
        sequence.append(self.END_TOKEN)

        for _ in range(desired_len - len(sequence)):
            sequence.append(self.PAD_TOKEN)

        return sequence

    def join_separated_sequences(self, first: List[Token], second: List[Token]) -> List[Token]:
        return [
            *first,
            self.SEPARATOR_TOKEN,
            *second
        ]
