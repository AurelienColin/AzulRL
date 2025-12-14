"""Bag container for drawing tiles in Azul."""
from dataclasses import dataclass
import numpy as np
from src.obj.container import Container
from src.obj.plate import Plate
from src.config import config


@dataclass
class Bag(Container):
    """Tile bag that holds all tiles not yet in play.

    At game start, contains 20 tiles of each color (100 total).
    Tiles are randomly drawn to fill factory plates each round.
    """
    n_blacks: int = config.n_tile_per_color
    n_whites: int = config.n_tile_per_color
    n_reds: int = config.n_tile_per_color
    n_blues: int = config.n_tile_per_color
    n_yellows: int = config.n_tile_per_color

    displayed_name: str = "Bag"

    def fill(self, graveyard: Container) -> None:
        """Refill the bag with tiles from the graveyard.

        Args:
            graveyard: Container holding discarded tiles to transfer.
        """
        self.n_blacks = graveyard.n_blacks
        self.n_whites = graveyard.n_whites
        self.n_reds = graveyard.n_reds
        self.n_blues = graveyard.n_blues
        self.n_yellows = graveyard.n_yellows

    def pick(self, index: int) -> Plate:
        """Randomly draw tiles from the bag to fill a plate.

        Args:
            index: Index to assign to the created plate.

        Returns:
            A Plate containing the randomly drawn tiles.

        Raises:
            AssertionError: If bag has fewer tiles than required per plate.
        """
        assert len(self) >= config.n_tile_per_plate, \
            f"Bag should contain at least {config.n_tile_per_plate} tiles but has {len(self)}"

        plate = Plate(index=index)
        for i in range(config.n_tile_per_plate):
            index = np.random.randint(len(self))
            if index < self.n_blacks:
                plate.n_blacks += 1
                self.n_blacks -= 1
            elif index < self.n_blacks + self.n_whites:
                plate.n_whites += 1
                self.n_whites -= 1
            elif index < self.n_blacks + self.n_whites + self.n_reds:
                plate.n_reds += 1
                self.n_reds -= 1
            elif index < self.n_blacks + self.n_whites + self.n_reds + self.n_blues:
                plate.n_blues += 1
                self.n_blues -= 1
            elif index < self.n_blacks + self.n_whites + self.n_reds + self.n_blues + self.n_yellows:
                plate.n_yellows += 1
                self.n_yellows -= 1
            else:
                raise ValueError(f"Index {index} out of bound (max = {len(self)})")
        return plate

    def __repr__(self) -> str:
        return super().__repr__()
