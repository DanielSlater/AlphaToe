"""
Full code for running a game of connect 4 on a board_width, board_height and winning length can be specified in relevant
methods. Allowing you to play connect 5, 6, 7, etc. Defaults are board_width = 7, board_height = 6, winning_length = 4

The main method to use here is play_game which simulates a game to the end using the function args it takes to determine
where each player plays.
The board is represented by a board_width x board_height tuple of ints. A 0 means no player has played in a space, 1
means player one has played there, -1 means the seconds player has played there. The apply_move method can be used to
return a copy of a given state with a given move applied. This can be useful for doing min-max or monte carlo sampling.
"""
import itertools
import random

from common.base_game_spec import BaseGameSpec
from games.tic_tac_toe_x import evaluate


def _new_board(board_width, board_height):
    """Return a emprty tic-tac-toe board we can use for simulating a game.

    Args:
        board_width (int): The width of the board, a board_width * board_height board is created
        board_height (int): The height of the board, a board_width * board_height board is created

    Returns:
        board_width x board_height tuple of ints
    """
    return tuple(tuple(0 for _ in range(board_height)) for _ in range(board_width))


def apply_move(board_state, move, side):
    """Returns a copy of the given board_state with the desired move applied.

    Args:
        board_state (2d tuple of int): The given board_state we want to apply the move to.
        move (int, int): The position we want to make the move in.
        side (int): The side we are making this move for, 1 for the first player, -1 for the second player.

    Returns:
        (2d tuple of int): A copy of the board_state with the given move applied for the given side.
    """
    move_x, move_y = move

    def get_tuples():
        for x in range(len(board_state)):
            if move_x == x:
                temp = list(board_state[x])
                temp[move_y] = side
                yield tuple(temp)
            else:
                yield board_state[x]

    return tuple(get_tuples())


def available_moves(board_state):
    """Get all legal moves for the current board_state. For Tic-tac-toe that is all positions that do not currently have
    pieces played.

    Args:
        board_state: The board_state we want to check for valid moves.

    Returns:
        Generator of (int, int): All the valid moves that can be played in this position.
    """
    for x, y in itertools.product(range(len(board_state)), range(len(board_state[0]))):
        if board_state[x][y] == 0:
            yield (x, y)


def _has_winning_line(line, winning_length):
    count = 0
    last_side = 0
    for x in line:
        if x == last_side:
            count += 1
            if count == winning_length:
                return last_side
        else:
            count = 1
            last_side = x
    return 0


def has_winner(board_state, winning_length=4):
    """Determine if a player has won on the given board_state.

    Args:
        board_state (2d tuple of int): The current board_state we want to evaluate.
        winning_length (int): The number of moves in a row needed for a win.

    Returns:
        int: 1 if player one has won, -1 if player 2 has won, otherwise 0.
    """
    board_width = len(board_state)
    board_height = len(board_state[0])

    # check rows
    for x in range(board_width):
        winner = _has_winning_line(board_state[x], winning_length)
        if winner != 0:
            return winner
    # check columns
    for y in range(board_height):
        winner = _has_winning_line((i[y] for i in board_state), winning_length)
        if winner != 0:
            return winner

    # check diagonals
    diagonals_start = -(board_width - winning_length)
    diagonals_end = (board_width - winning_length)
    for d in range(diagonals_start, diagonals_end+1):
        winner = _has_winning_line(
            (board_state[i][i + d] for i in range(max(-d, 0), min(board_width, board_height - d))),
            winning_length)
        if winner != 0:
            return winner
    for d in range(diagonals_start, diagonals_end+1):
        winner = _has_winning_line(
            (board_state[i][board_height - i - d - 1] for i in range(max(-d, 0), min(board_width, board_height - d))),
            winning_length)
        if winner != 0:
            return winner

    return 0  # no one has won, return 0 for a draw


def play_game(plus_player_func, minus_player_func, board_width=7, board_height=6, winning_length=4, log=False):
    """Run a single game of tic-tac-toe until the end, using the provided function args to determine the moves for each
    player.

    Args:
        plus_player_func ((board_state(board_size by board_size tuple of int), side(int)) -> move((int, int))):
            Function that takes the current board_state and side this player is playing, and returns the move the player
            wants to play.
        minus_player_func ((board_state(board_size by board_size tuple of int), side(int)) -> move((int, int))):
            Function that takes the current board_state and side this player is playing, and returns the move the player
            wants to play.
        board_width (int): The width of the board, a board_width * board_height board is created
        board_height (int): The height of the board, a board_width * board_height board is created
        winning_length (int): The number of pieces in a row needed to win a game.
        log (bool): If True progress is logged to console, defaults to False

    Returns:
        int: 1 if the plus_player_func won, -1 if the minus_player_func won and 0 for a draw
    """
    board_state = _new_board(board_width, board_height)
    player_turn = 1

    while True:
        _avialable_moves = list(available_moves(board_state))
        if len(_avialable_moves) == 0:
            # draw
            if log:
                print("no moves left, game ended a draw")
            return 0.
        if player_turn > 0:
            move = plus_player_func(board_state, 1)
        else:
            move = minus_player_func(board_state, -1)

        if move not in _avialable_moves:
            # if a player makes an invalid move the other player wins
            if log:
                print("illegal move ", move)
            return -player_turn

        board_state = apply_move(board_state, move, player_turn)
        if log:
            print(board_state)

        winner = has_winner(board_state, winning_length)
        if winner != 0:
            if log:
                print("we have a winner, side: %s" % player_turn)
            return winner
        player_turn = -player_turn


def random_player(board_state, _):
    """A player func that can be used in the play_game method. Given a board state it chooses a move randomly from the
    valid moves in the current state.

    Args:
        board_state (2d tuple of int): The current state of the board
        _: the side this player is playing, not used in this function because we are simply choosing the moves randomly

    Returns:
        (int, int): the move we want to play on the current board
    """
    moves = list(available_moves(board_state))
    return random.choice(moves)


class Connect4GameSpec(BaseGameSpec):
    def __init__(self, board_width=7, board_height=6, winning_length=4):
        self._board_height = board_height
        self._board_width = board_width
        self._winning_length = winning_length
        self.available_moves = available_moves
        self.apply_move = apply_move

    def new_board(self):
        return _new_board(self._board_width, self._board_height)

    def has_winner(self, board_state):
        return has_winner(board_state, self._winning_length)

    def board_dimensions(self):
        return self._board_width, self._board_height

    def flat_move_to_tuple(self, move_index):
        move_x = move_index % self._board_width
        move_y = int(move_index / self._board_width)
        return (move_x, move_y)

    def outputs(self):
        return self._board_width * self._board_height

    def evaluate(self, board_state):
        return evaluate(board_state, self._winning_length)

if __name__ == '__main__':
    # example of playing a game
    play_game(random_player, random_player, log=True, board_width=7, board_height=6, winning_length=4)
