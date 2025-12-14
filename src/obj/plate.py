"""Plate container for Azul game tiles."""
from dataclasses import dataclass
from src.obj.container import Container


@dataclass
class Plate(Container):
    """Factory plate that holds tiles to be selected by players.

    Each plate initially receives tiles from the bag at the start of a round.
    When a player selects tiles from a plate, remaining tiles go to the central.
    """
    def __post_init__(self):
        self.displayed_name = f"Plate {self.index}"


    def empty(self) -> None:
        self.n_blacks = self.n_whites = self.n_reds = self.n_blues = self.n_yellows = 0


    def __repr__(self) -> str:
        return super().__repr__()
