from src.obj.container import Container
from dataclasses import dataclass
import typing


@dataclass
class Central(Container):
    has_first_player_tile = True

    displayed_name: str = "Center"

    def __repr__(self) -> str:
        description = "\n".join((
            super().__repr__(),
            f"Has first player tile: {self.has_first_player_tile}"
        ))
        return description
