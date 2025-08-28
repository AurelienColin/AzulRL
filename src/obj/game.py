from dataclasses import dataclass
import typing
import numpy as np

from src.config import config
from src.obj.player import Player
from src.obj.plate import Plate
from src.obj.bag import Bag
from rignak.src.lazy_property import LazyProperty
from src.obj.container import Container
from src.obj.central import Central
import sys

@dataclass
class Game:
    n_players: int = 4

    _bag: typing.Optional[Bag] = None
    _plates: typing.Optional[typing.List[Plate]] = None
    _central: typing.Optional[Central] = None
    _graveyard: typing.Optional[Container] = None
    _players: typing.Optional[typing.List[Player]] = None

    def __post_init__(self):
        self._plates = self.get_plates()

    @property
    def n_plates(self) -> int:
        return config.get_plate_number(self.n_players)

    @LazyProperty
    def bag(self) -> Bag:
        return Bag()

    @LazyProperty
    def plates(self) -> typing.List[Plate]:
        raise ValueError(f"`get_plates` should have been called before accessing self.plates.")

    def get_plates(self) -> typing.List[Plate]:
        plates = [Plate(index=i) for i in range(self.n_plates)]
        for i, plate in enumerate(plates):
            if len(self.bag) < config.n_tile_per_plate:
                self._bag = Bag()
                # if len(self.graveyard) < config.n_tile_per_plate:
                #     break
                #
                # self.bag.fill(self.graveyard)
                # print(f"Atfter filling the bag: {len(self.bag)=}")
                self.graveyard.empty()

            plates[i] = self.bag.pick(i)
        self.central.has_first_player_tile = True
        return plates

    @LazyProperty
    def graveyard(self) -> Container:
        return Container(displayed_name="Graveyard")

    @LazyProperty
    def central(self) -> Central:
        return Central()

    @LazyProperty
    def players(self) -> typing.List[Player]:
        players = [Player(index=i) for i in range(self.n_players)]
        players[0].is_first = True
        return players

    def round(self, printing=True):
        choices = [[] for _ in range(self.n_players)]
        states = [[] for _ in range(self.n_players)]

        self._plates = self.get_plates()

        start = False

        player_index: int = 0

        ns = [len(plate) for plate in (*self.plates, self.central)]
        print(f"{ns}")
        if sum(ns) == 0:
            sys.exit()

        for player in self.players:
            player.penalties.n = 0

        table_turn = 0
        while True:
            if all(not len(plate) for plate in (*self.plates, self.central)):
                print(f"Empty plates.")
                break

            player = self.players[player_index]
            if not start:
                if not any((player.is_first for player in self.players)):
                    sys.exit()

                if not player.is_first:
                    player_index = (player_index + 1) % self.n_players
                    continue
                else:
                    start = True
                    player.is_first = False

            if printing:
                print(self)
            state = self.get_state()
            i_plate, i_color, i_row = player.choose(state)
            if printing:
                print(f"Player {player.index} choose plate {i_plate}, color {i_color} and put it in row {i_row}")
            choices[player_index].append((i_plate, i_color, i_row))
            states[player_index].append(state)

            if i_plate == config.n_plates:
                self.central[i_color] = 0
                if self.central.has_first_player_tile:
                    player.is_first = True
                    player.penalties.n += 1
                    self.central.has_first_player_tile = False
            else:
                for j_color in range(config.n_colors):
                    if j_color != i_color:
                        self.central[j_color] += self.plates[i_plate][j_color]
                    self.plates[i_plate][j_color] = 0
            player_index = (player_index + 1) % self.n_players
        return choices, states

    def has_ended(self) -> bool:
        return any(player.right.is_complete for player in self.players)

    def end_of_round(self, printing=True):
        if printing:
            print(self)

        choices = []
        states = []
        for i, player in enumerate(self.players):
            state = self.get_state()
            choice = player.end_of_round(state)
            choices.append(choice)
            states.append(state)
        return choices, states

    def get_state(self) -> np.ndarray:
        """
        The game state is composed of:
        - [0->n_colors]: the number of tiles in the bag for each color.
        - [n_colors -> 2*n_colors]: the number of tiles in the graveyard for each color.
        - [2*n_colors -> (2 + n_plates) * n_colors]: the number of tiles in the plates for each color.
        - [(2 + n_plates) * n_colors -> (3 + n_plates) * n_colors]: the number of tiles in the central for each color.
        - [(3 + n_plates) * n_colors]: 1 if the First Player Token still in the central.
        - [(3 + n_plates) * n_colors +1 -> -1]: the state of each player.
            [... -> ... + n_colors]: The number of tiles in each row of the left panel.
            [... + n_colors -> ... + 2*n_colors]: The color of the tile in each row of the left panel.
            [... + 2*n_colors -> ... + 2*n_colors + n_colors**2]: The color of the tile in each tile of the right panel.
            [... + 2*n_colors + n_colors**2]: The number of penalties.
        """
        n_state_per_container = config.n_colors
        n_states = (3 + self.n_plates) * n_state_per_container + 1 + \
                   self.n_players * self.players[0].n_states

        states = np.zeros(n_states, dtype=int)

        index = 0
        for container in (self.bag, self.graveyard, *self.plates, self.central):
            for i in range(config.n_colors):
                states[index] = container[i]
                index += 1

        states[index] = int(self.central.has_first_player_tile)
        index += 1

        for player in self.players:
            states[index: index + player.n_states] = player.get_state()
            index += player.n_states
        return states

    def run(self) -> None:
        while True:
            self.round()
            self.end_of_round()

            if self.has_ended():
                for player in self.players:
                    print(f"{player.prefix}score: {player.score} pts.")

    def __repr__(self) -> str:
        representation = "\n\n".join(
            (
                '-' * 20,
                repr(self.bag),
                repr(self.graveyard),
                repr(self.central),
                *(repr(plate) for plate in self.plates),
                *(repr(player) for player in self.players),
            )
        )
        return representation
