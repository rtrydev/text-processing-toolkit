from abc import ABC, abstractmethod
from typing import List

from toolkit.classes.token import Token


class BaseFileReader(ABC):
    @abstractmethod
    def tokenize_file(self, file_path: str, line_limit: int = -1) -> List[List[Token]]:
        pass
