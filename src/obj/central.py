"""Central area container for Azul game tiles."""
from dataclasses import dataclass
from src.obj.container import Container


@dataclass
class Central(Container):
    """Central area where remaining tiles accumulate during a round.

    Tiles that are not selected from factory plates move to the central.
    Also tracks the first player tile which gives a penalty but grants
    first turn next round.
    """

    has_first_player_tile: bool = True

    displayed_name: str = "Center"

    def __repr__(self) -> str:
        description = "\n".join((
            super().__repr__(),
            f"Has first player tile: {self.has_first_player_tile}"
        ))
        return description
