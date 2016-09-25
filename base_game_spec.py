import operator
import random
from functools import reduce


class BaseGameSpec(object):
    def __init__(self):
        raise NotImplementedError('This is an abstract base class')

    def new_board(self):
        raise NotImplementedError()

    def apply_move(self):
        raise NotImplementedError()

    def available_moves(self):
        raise NotImplementedError()

    def has_winner(self, board_sate):
        raise NotImplementedError()

    def board_dimensions(self):
        """Returns the dimensions of the board for this game

        Returns:
            tuple of ints: one int for each dimension of the board, this will normally be 2 ints
        """
        raise NotImplementedError()

    def board_squares(self):
        return reduce(operator.mul, self.board_dimensions(), 1)

    def outputs(self):
        """The number of moves that could be made in this kind of game, weather or not they are legal. For most games
        this will be every single square on the board, but for connect 4 this is different. If we wanted to do chess in
        the future this method may need to get a bit more complicated.

        Returns:
            int
        """
        return self.board_squares()

    def flat_move_to_tuple(self, move_index):
        board_x = self.board_dimensions()[0]
        return move_index / board_x, move_index % board_x

    def play_game(self, plus_player_func, minus_player_func, log=False):
        """Run a single game of until the end, using the provided function args to determine the moves for each
        player.

        Args:
            plus_player_func ((board_state(3 by 3 tuple of int), side(int)) -> move((int, int))): Function that takes the
                current board_state and side this player is playing, and returns the move the player wants to play.
            minus_player_func ((board_state(3 by 3 tuple of int), side(int)) -> move((int, int))): Function that takes the
                current board_state and side this player is playing, and returns the move the player wants to play.
            log (bool): If True progress is logged to console, defaults to False

        Returns:
            int: 1 if the plus_player_func won, -1 if the minus_player_func won and 0 for a draw
        """
        board_state = self.new_board()
        player_turn = 1

        while True:
            _available_moves = list(self.available_moves(board_state))

            if len(_available_moves) == 0:
                # draw
                if log:
                    print("no moves left, game ended a draw")
                return 0.
            if player_turn > 0:
                move = plus_player_func(board_state, 1)
            else:
                move = minus_player_func(board_state, -1)

            if move not in _available_moves:
                # if a player makes an invalid move the other player wins
                if log:
                    print("illegal move ", move)
                return -player_turn

            board_state = self.apply_move(board_state, move, player_turn)
            if log:
                print(board_state)

            winner = self.has_winner(board_state)
            if winner != 0:
                if log:
                    print("we have a winner, side: %s" % player_turn)
                return winner
            player_turn = -player_turn

    def get_random_player_func(self):
        """Return a function that makes moves for the current game by choosing a move randomly
        NOTE: this move returns the function that makes the random move so should be used like so:
        Examples:
            self.play_game(self.get_random_player_func(), self.get_random_player_func())

        Returns:
            board_state, side (int) -> move : function that plays this game by making random moves
        """
        return lambda board_state, side: random.choice(list(self.available_moves(board_state)))