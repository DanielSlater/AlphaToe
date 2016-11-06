from unittest import TestCase

from games.tic_tac_toe import TicTacToeGameSpec
from techniques.min_max import min_max_alpha_beta, min_max


class TestMinMax(TestCase):
    def setUp(self):
        self._game_spec = TicTacToeGameSpec()

    def test_basic_position(self):
        # the best move is 2, 2 forcing a win with pluses next move, both players should select it
        board_state = ((0, 0, 0),
                       (-1, -1, 1),
                       (1, 0, 0))

        result_min_max = min_max(self._game_spec, board_state, 1, 8)
        result_min_max_alpha_beta = min_max_alpha_beta(self._game_spec, board_state, 1, 8)

        self.assertEqual(result_min_max[1], (2, 2))
        self.assertEqual(result_min_max_alpha_beta[1], (2, 2))

    def test_basic_position_for_minus_player(self):
        board_state = ((-1, 1, 0),
                       (1, -1, 1),
                       (1, 0, 0))

        result_min_max = min_max(self._game_spec, board_state, -1, 8)
        result_min_max_alpha_beta = min_max_alpha_beta(self._game_spec, board_state, -1, 8)

        self.assertEqual(result_min_max[1], (2, 2))
        self.assertEqual(result_min_max_alpha_beta[1], (2, 2))
