from unittest import TestCase

from games.tic_tac_toe import TicTacToeGameSpec
from techniques.create_positions_set import create_positions_set


class TestCreatePositionsSet(TestCase):
    def setUp(self):
        self._game_spec = TicTacToeGameSpec()

    def test_create_positions(self):
        number_of_positions = 100
        positions = create_positions_set(self._game_spec, number_of_positions, self._game_spec.get_random_player_func())

        self.assertGreater(len(positions), number_of_positions-1)