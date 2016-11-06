"""
For games like tic-tac-toe we are unlikely to be able to find databases of top level games. This file allows us
generate sets games using good existing algorithms from which we can train our networks. Such as min max
"""
import pickle
import random

import zlib

from common.network_helpers import invert_board_state
from techniques.min_max import min_max_alpha_beta


def create_positions_set(game_spec, number_of_positions, choose_move_func, compress=False):
    """Generate a set of positions. All positions are set to be from the point of view of the plus player. In order to
    aid breadth of search if a position that we have already calculated the best move for comes up twice we choose a
    random move. Moves chosen randomly are not stored in the returned set.

    Args:
        game_spec (common.BaseGameSpec):
        number_of_positions (int): We will simulate this many positions
        choose_move_func (): Function that picks the best move from a board position
        compress (bool): If True then we will compress all the state pairs we store to save on memory, use
            pickle.loads(zlib.decompress(item)) to uncompress

    Returns:
        {board_state, move}
    """
    positions = {}
    random_player_func = game_spec.get_random_player_func()

    def store_move_pair(board_state, side):
        if side != 1:
            board_state_for_plus = invert_board_state(board_state)
        else:
            board_state_for_plus = board_state

        if compress:
            board_state_for_plus = zlib.compress(pickle.dumps(board_state_for_plus))

        # if we have already seen this position then make a random move to increase position diversity
        if board_state_for_plus in positions:
            return random_player_func(board_state, side)
        else:
            move = choose_move_func(board_state, side)
            positions[board_state_for_plus] = move

            return move

    while number_of_positions > len(positions.keys()):
        game_spec.play_game(store_move_pair, store_move_pair)
        print(len(positions.keys()))

    return positions


if __name__ == '__main__':
    # example usage
    from games.connect_4 import Connect4GameSpec

    game_spec = Connect4GameSpec()

    def choose_move_func(board_state, side):
        return min_max_alpha_beta(game_spec, board_state, side, 6)[1]

    positions = create_positions_set(game_spec, 10000, choose_move_func)

    positions_as_array = [(x, y) for x, y in positions.items()]
    random.shuffle(positions_as_array)

    with open('position_connect_4_min_max_depth_6', mode='wb') as f:
        pickle.dump(positions_as_array, f)

    print("created")