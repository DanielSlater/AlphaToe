import operator
import random
from functools import reduce


class BaseGameSpec(object):
    def __init__(self):
        """Abstract base class for the specification for running/training on a game.

        Examples:
            spec = TicTacToeGameSpec()
            result = spec.play_game(func_a, func_b)
        """
        raise NotImplementedError('This is an abstract base class')

    def new_board(self):
        raise NotImplementedError()

    def apply_move(self, board_state, move, side):
        raise NotImplementedError()

    def available_moves(self, board_state):
        raise NotImplementedError()

    def has_winner(self, board_state):
        raise NotImplementedError()

    def evaluate(self, board_state):
        """An evaluation function for this game, gives an estimate of how good the board position is for the plus player.
        There is no specific range for the values returned, they just need to be relative to each other.

        Args:
            board_state (tuple): State of the board

        Returns:
            number
        """
        raise NotImplementedError()

    def board_dimensions(self):
        """Returns the dimensions of the board for this game

        Returns:
            tuple of ints: one int for each dimension of the board, this will normally be 2 ints
        """
        raise NotImplementedError()

    def board_squares(self):
        """The number of squares on the board. This can be used for the number of input nodes to a network.

        Returns:
            int
        """
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
        """If board is 2d then we return a tuple for where we moved to.
        e.g if the board is a 3x3 size and our move_index was 6 then
        this method will return (2, 0)

        Args:
            move_index (int): The index of the square we moved to

        Returns:
            tuple or int: For where we moved in board coordinates
        """
        if len(self.board_dimensions()) == 1:
            return move_index

        board_x = self.board_dimensions()[0]
        return int(move_index / board_x), move_index % board_x

    def tuple_move_to_flat(self, tuple_move):
        """Does the inverse operation to flat_move_to_tuple

        Args:
            tuple_move (tuple):

        Returns:
            int :
        """
        if len(self.board_dimensions()) == 1:
            return tuple_move[0]
        else:
            return tuple_move[0] * self.board_dimensions()[0] + tuple_move[1]

    def play_game(self, plus_player_func, minus_player_func, log=False, board_state=None):
        """Run a single game of until the end, using the provided function args to determine the moves for each
        player.

        Args:
            plus_player_func ((board_state(3 by 3 tuple of int), side(int)) -> move((int, int))): Function that takes the
                current board_state and side this player is playing, and returns the move the player wants to play.
            minus_player_func ((board_state(3 by 3 tuple of int), side(int)) -> move((int, int))): Function that takes the
                current board_state and side this player is playing, and returns the move the player wants to play.
            log (bool): If True progress is logged to console, defaults to False
            board_state: Optionally have the game start from this position, rather than from a new board

        Returns:
            int: 1 if the plus_player_func won, -1 if the minus_player_func won and 0 for a draw
        """
        board_state = board_state or self.new_board()
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
