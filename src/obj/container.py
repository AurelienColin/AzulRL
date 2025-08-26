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

    displayed_name: str = "Undefined"

    def empty(self) -> None:
        self.n_blacks: int = 0
        self.n_whites: int = 0
        self.n_reds: int = 0
        self.n_blues: int = 0
        self.n_yellows: int = 0

    def __len__(self) -> int:
        return self.n_blacks + self.n_whites + self.n_reds + self.n_blues + self.n_yellows

    def __setitem__(self, key: int, value: int) -> None:
        if key == 0:
            self.n_blacks = value
        elif key == 1:
            self.n_whites = value
        elif key == 2:
            self.n_reds = value
        elif key == 3:
            self.n_blues = value
        elif key == 4:
            self.n_yellows = value

    def __getitem__(self, item: int) -> int:
        return (self.n_blacks, self.n_whites, self.n_reds, self.n_blues, self.n_yellows)[item]

    def __repr__(self) -> str:
        description = '\n'.join((
            f'-- {self.displayed_name} --',
            *(f"{i} - {color} tiles: {self[i]}" for i, color in enumerate(config.colors))
        ))
        return description
