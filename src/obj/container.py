from dataclasses import dataclass
import typing
from src.config import config

@dataclass
class Container:
    index: int = 0
    
    n_blacks: int = 0
    n_whites: int = 0
    n_reds: int = 0
    n_blues: int = 0
    n_yellows: int = 0

    displayed_name: int = "Undefined"

    def empty(self) -> None:
        self.n_blacks: int = 0
        self.n_whites: int = 0
        self.n_reds: int = 0
        self.n_blues: int = 0
        self.n_yellows: int = 0

    def __len__(self) -> int:
        return self.n_blacks + self.n_whites + self.n_reds + self.n_blues + self.n_yellows

    def __getitem__(self, item: int) -> int:
        return (self.n_blacks, self.n_whites, self.n_reds, self.n_blues, self.n_yellows)[item]

    def __repr__(self) -> str:
        description = '\n'.join((
            '-- {self.displayed_name} --',
            *(f"{color} tiles: {self[i]}" for i, color in enumerate(config.colors))
        ))
        return description