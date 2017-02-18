from unittest import TestCase

from games.connect_4 import has_winner, play_game, random_player


class TestTicTacToeX(TestCase):
    def test_has_winner(self):
        board_state = ((0, 0, 0, 0, 0, 0),
                       (0, 0, 0, 0, 0, 0),
                       (0, 0, 0, 0, 0, 0),
                       (1, 0, 0, 0, 0, 0),
                       (0, 1, 0, 0, 0, 0),
                       (0, 0, 1, 0, 0, 0),
                       (0, 0, 0, 1, 0, 0))

        self.assertEqual(1, has_winner(board_state), 4)

        board_state = ((0, 0, 0, 0, 1, 0),
                       (0, 0, 0, 1, 0, 0),
                       (0, 0, 1, 0, 0, 0),
                       (0, 1, 0, 0, 0, 0),
                       (0, 0, 0, 0, 0, 0),
                       (0, 0, 0, 0, 0, 0),
                       (0, 0, 0, 0, 0, 0))

        self.assertEqual(1, has_winner(board_state), 4)

    def test_play_game(self):
        play_game(random_player, random_player)