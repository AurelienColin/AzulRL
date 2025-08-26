from dataclasses import dataclass
import typing
from src.obj.container import Container

@dataclass
class Plate(Container):
    def __post_init__(self):
        self.displayed_name = f"Plate {self.index}"


    def empty(self) -> None:
        self.n_blacks = self.n_whites = self.n_reds = self.n_blues = self.n_yellows = 0


    def __repr__(self) -> str:
        return super().__repr__()
