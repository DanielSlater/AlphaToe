from unittest import TestCase

from games.tic_tac_toe_x import has_winner, _has_winning_line, play_game, random_player, evaluate


class TestTicTacToeX(TestCase):

    def test_has_winning_line(self):
        self.assertEqual(1, _has_winning_line((0, 1, 1, 1, 1), 4))
        self.assertEqual(0, _has_winning_line((0, 1, -1, 1, 1), 4))
        self.assertEqual(1, _has_winning_line((1, 1, 1, 1, 1, 0), 4))
        self.assertEqual(1, _has_winning_line((1, 0, 1, 1, 1, 0), 3))
        self.assertEqual(-1, _has_winning_line((-1, -1, -1, -1, 1), 4))

    def test_has_winner(self):
        board_state = ((0, 0, 0, 0, 0),
                       (0, -1, 0, 0, 0),
                       (0, -1, 0, 0, 0),
                       (0, -1, 0, 0, 0),
                       (0, -1, 0, 0, 0))
        self.assertEqual(-1, has_winner(board_state, 4))

        board_state = ((0, 1, 0, 0, 0),
                       (0, 0, 1, 0, 0),
                       (0, 0, 0, 1, 0),
                       (0, 0, 0, 0, 1),
                       (0, 0, 0, 0, 0))
        self.assertEqual(1, has_winner(board_state, 4))

        board_state = ((0, 0, 0, 0, 0),
                       (0, 0, 0, 0, 1),
                       (0, 0, 0, 1, 0),
                       (0, 0, 1, 0, 0),
                       (0, 1, 0, 0, 0))
        self.assertEqual(1, has_winner(board_state, 4))

        board_state = ((0, 0, 0, -1, 0),
                       (0, 0, -1, 0, 0),
                       (0, -1, 0, 0, 0),
                       (-1, 0, 0, 0, 0),
                       (0, 0, 0, 0, 0))
        self.assertEqual(-1, has_winner(board_state, 4))

    def test_play_game(self):
        play_game(random_player, random_player)

    def test_has_evaluate(self):
        board_state = ((-1, -1, -1, 0, 0),
                       (0, 0, 0, 0, 0),
                       (0, -1, 0, 0, 0),
                       (0, -1, 0, 0, 0),
                       (0, -1, 0, 0, 0))
        self.assertGreater(0, evaluate(board_state, 4))

        board_state = ((0, 1, 0, 0, 0),
                       (0, 0, 1, 0, 0),
                       (0, 0, 0, 0, 0),
                       (0, 0, 0, 0, 1),
                       (0, 0, 0, 0, 0))
        self.assertGreater(evaluate(board_state, 4), 0)

        board_state = ((0, 0, 0, 0, 0),
                       (0, 0, 0, 0, 1),
                       (0, 0, 0, 0, 0),
                       (0, 0, 1, 0, 0),
                       (0, 1, 1, 0, 0))
        self.assertGreater(evaluate(board_state, 4), 0)

        board_state = ((0, 0, 0, -1, 0),
                       (0, 0, 0, 0, 0),
                       (0, 0, 0, 0, 0),
                       (-1, 0, 0, 0, 0),
                       (0, 0, 0, 0, 0))
        self.assertGreater(0, evaluate(board_state, 4))