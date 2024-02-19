from abc import ABC, abstractmethod
from typing import List

from toolkit.classes.token import Token


class BaseTokenizer(ABC):
    @abstractmethod
    def tokenize_string(self, line: str) -> List[Token]:
        pass

