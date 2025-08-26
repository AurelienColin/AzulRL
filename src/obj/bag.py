from dataclasses import dataclass
import typing
import numpy as np
from src.obj.container import Container
from src.obj.plate import Plate
from src.config import config


@dataclass
class Bag(Container):
    n_blacks: int = config.n_tile_per_color
    n_whites: int = config.n_tile_per_color
    n_reds: int = config.n_tile_per_color
    n_blues: int = config.n_tile_per_color
    n_yellows: int = config.n_tile_per_color

    displayed_name: str = "Bag"

    def fill(self, graveyard: Container) -> None:
        self.n_blacks = graveyard.n_blacks
        self.n_whites = graveyard.n_whites
        self.n_reds = graveyard.n_reds
        self.n_blues = graveyard.n_blues
        self.n_yellows = graveyard.n_yellows

    def pick(self) -> int:
        assert len(self) >= config.n_tile_per_plate, \
            f"Bag should contain at least {config.n_tile_per_plate} tiles but has {len(self)}"

        plate = Plate()
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
                plate.n_reds += 1
                self.n_reds -= 1
            elif index < self.n_blacks + self.n_whites + self.n_reds + self.n_blues + self.n_yellows:
                plate.n_yellows += 1
                self.n_yellows -= 1
            else:
                raise ValueError(f"Index {index} out of bound (max = {len(self)})")
        return plate
