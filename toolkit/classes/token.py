from dataclasses import dataclass
from typing import Dict


@dataclass
class Token:
    body: str
    data: Dict
