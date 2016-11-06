import sys


def _score_line(line):
    minus_count = line.count(-1)
    plus_count = line.count(1)
    if minus_count + plus_count < 3:
        if minus_count == 2:
            return -1
        elif plus_count == 2:
            return 1
    return 0


def evaluate_tic_tac_toe(board_state):
    """Get a rough score for how good we think this board position is for the plus_player for the game tic-tac-toe. Does
    this based on number of 2 in row lines we have.

    Args:
        board_state (3x3 tuple of int): The board state we are evaluating

    Returns:
        int: evaluated score for the position for the plus player, posative is good for the plus player, negative good
            for the minus player
    """
    score = 0
    for x in range(3):
        score += _score_line(board_state[x])
    for y in range(3):
        score += _score_line([i[y] for i in board_state])

    # diagonals
    score += _score_line([board_state[i][i] for i in range(3)])
    score += _score_line([board_state[2 - i][i] for i in range(3)])

    return score


def min_max(game_spec, board_state, side, max_depth, evaluation_func=None):
    """Runs the min_max_algorithm on a given board_sate for a given side, to a given depth in order to find the best
    move

    Args:
        game_spec (BaseGameSpec): The specification for the game we are evaluating
        evaluation_func (board_state -> int): Function used to evaluate the position for the plus player, If None then
            we will use the evaluation function from the game_spec
        board_state (3x3 tuple of int): The board state we are evaluating
        side (int): either +1 or -1
        max_depth (int): how deep we want our tree to go before we use the evaluate method to determine how good the
        position is.

    Returns:
        (best_score(int), best_score_move((int, int)): the move found to be best and what it's min-max score was
    """
    best_score = None
    best_score_move = None
    evaluation_func = evaluation_func or game_spec.evaluate

    moves = list(game_spec.available_moves(board_state))
    if not moves:
        # this is a draw
        return 0, None

    for move in moves:
        new_board_state = game_spec.apply_move(board_state, move, side)
        winner = game_spec.has_winner(new_board_state)
        if winner != 0:
            return winner * 10000, move
        else:
            if max_depth <= 1:
                score = evaluation_func(new_board_state)
            else:
                score, _ = min_max(game_spec, new_board_state, -side, max_depth - 1, evaluation_func=evaluation_func)
            if side > 0:
                if best_score is None or score > best_score:
                    best_score = score
                    best_score_move = move
            else:
                if best_score is None or score < best_score:
                    best_score = score
                    best_score_move = move
    return best_score, best_score_move


def min_max_alpha_beta(game_spec, board_state, side, max_depth, evaluation_func=None, alpha=-sys.float_info.max,
                       beta=sys.float_info.max):
    """Runs the min_max_algorithm on a given board_sate for a given side, to a given depth in order to find the best
    move

    Args:
        game_spec (BaseGameSpec): The specification for the game we are evaluating
        evaluation_func (board_state -> int): Function used to evaluate the position for the plus player
        board_state (3x3 tuple of int): The board state we are evaluating
        side (int): either +1 or -1
        max_depth (int): how deep we want our tree to go before we use the evaluate method to determine how good the
        position is.
        alpha (float): Used when this is called recursively, normally ignore
        beta (float): Used when this is called recursively, normally ignore

    Returns:
        (best_score(int), best_score_move((int, int)): the move found to be best and what it's min-max score was
    """
    evaluation_func = evaluation_func or game_spec.evaluate
    best_score_move = None
    moves = list(game_spec.available_moves(board_state))
    if not moves:
        return 0, None

    for move in moves:
        new_board_state = game_spec.apply_move(board_state, move, side)
        winner = game_spec.has_winner(new_board_state)
        if winner != 0:
            return winner * 10000, move
        else:
            if max_depth <= 1:
                score = evaluation_func(new_board_state)
            else:
                score, _ = min_max_alpha_beta(game_spec, new_board_state, -side, max_depth - 1, evaluation_func, alpha,
                                              beta)

        if side > 0:
            if score > alpha:
                alpha = score
                best_score_move = move
        else:
            if score < beta:
                beta = score
                best_score_move = move
        if alpha >= beta:
            break

    return alpha if side > 0 else beta, best_score_move


def min_max_player(board_state, side):
    return min_max(board_state, side, 5)[1]


def evaluate(board_state):
    """Get a rough score for how good we think this board position is for the plus_player. Does this based on number of
    2 in row lines we have.

    Args:
        board_state (3x3 tuple of int): The board state we are evaluating

    Returns:
        int: evaluated score for the position for the plus player, posative is good for the plus player, negative good
            for the minus player
    """
    score = 0
    for x in range(len(board_state)):
        score += _score_line(board_state[x])
    for y in range(len(board_state[0])):
        score += _score_line([i[y] for i in board_state])

    # diagonals
    score += _score_line([board_state[i][i] for i in range(3)])
    score += _score_line([board_state[2 - i][i] for i in range(3)])

    return score
