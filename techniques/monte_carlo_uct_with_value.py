import collections
import random

import math

from techniques.monte_carlo import _upper_confidence_bounds


def monte_carlo_tree_search_uct_with_value(game_spec, board_state, side, number_of_samples, value_func,
                                           value_weighting):
    """Evaluate the best from the current board_state for the given side using monte carlo sampling with upper
    confidence bounds for trees.

    Args:
        game_spec (BaseGameSpec): The specification for the game we are evaluating
        board_state (3x3 tuple of int): state of the board
        side (int): side currently to play. +1 for the plus player, -1 for the minus player
        number_of_samples (int): number of samples rollouts to run from the current position, the higher the number the
            better the estimation of the position
        value_func (board_state, side -> float):
        value_weighting (float): parameter to adjust how much priority we give to the value_func

    Returns:
        (result(int), move(int,int)): The average result for the best move from this position and what that move was.
    """
    state_results = collections.defaultdict(float)
    state_samples = collections.defaultdict(float)
    state_values = collections.defaultdict(float)

    for _ in range(number_of_samples):
        current_side = side
        current_board_state = board_state
        first_unvisited_node = True
        rollout_path = []
        result = 0

        while result == 0:
            move_states = {move: game_spec.apply_move(current_board_state, move, current_side)
                           for move in game_spec.available_moves(current_board_state)}

            if not move_states:
                result = 0
                break

            if all((state in state_samples) for _, state in move_states):
                log_total_samples = math.log(sum(state_samples[s] for s in move_states.values()))
                move, state = max(move_states, key=lambda _, s: state_values[s] * value_weighting +
                                                                _upper_confidence_bounds(state_results[s],
                                                                                         state_samples[s],
                                                                                         log_total_samples))
            else:
                move = random.choice(list(move_states.keys()))

            current_board_state = move_states[move]

            if first_unvisited_node:
                rollout_path.append((current_board_state, current_side))
                if current_board_state not in state_samples:
                    state_values[current_board_state] = value_func(current_board_state)
                    first_unvisited_node = False

            current_side = -current_side

            result = game_spec.has_winner(current_board_state)

        for path_board_state, path_side in rollout_path:
            state_samples[path_board_state] += 1.
            result *= path_side
            # normalize results to be between 0 and 1 before this it between -1 and 1
            result /= 2.
            result += .5
            state_results[path_board_state] += result

    move_states = {move: game_spec.apply_move(board_state, move, side) for move in
                   game_spec.available_moves(board_state)}

    move = max(move_states, key=lambda x: state_results[move_states[x]] / state_samples[move_states[x]])

    return state_results[move_states[move]] / state_samples[move_states[move]], move
