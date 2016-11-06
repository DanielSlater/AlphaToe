import operator
import pickle
from functools import reduce

import numpy as np
import tensorflow as tf


def create_network(input_nodes, hidden_nodes, output_nodes=None, output_softmax=True):
    """Create a network with relu activations at each layer

    Args:
        output_nodes: (int): Number of output nodes, if None then number of input nodes is used
        input_nodes (int or tuple(int)): The size of the board this network will work on. The output layer will also be
            this size if not specified. Can be an int if 1d or a tuple of ints for a 2d+ dim board
        hidden_nodes ([int]): The number of hidden nodes in each hidden layer
        output_softmax (bool): If True softmax is used in the final layer, otherwise just use the activation with no
            non-linearity function

    Returns:
        (input_layer, output_layer, [variables]) : The final item in the tuple is a list containing all the parameters,
            wieghts and biases used in this network
    """
    output_nodes = output_nodes or input_nodes

    variables = []

    with tf.name_scope('network'):
        if isinstance(input_nodes, tuple):
            input_layer = tf.placeholder("float", (None,) + input_nodes)
            flat_size = reduce(operator.mul, input_nodes, 1)
            current_layer = tf.reshape(input_layer, (-1, flat_size))
        else:
            input_layer = tf.placeholder("float", (None, input_nodes))
            current_layer = input_layer

        for hidden_nodes in hidden_nodes:
            last_layer_nodes = int(current_layer.get_shape()[-1])
            hidden_weights = tf.Variable(
                tf.truncated_normal((last_layer_nodes, hidden_nodes), stddev=1. / np.sqrt(last_layer_nodes)),
                name='weights')
            hidden_bias = tf.Variable(tf.constant(0.01, shape=(hidden_nodes,)), name='biases')

            variables.append(hidden_weights)
            variables.append(hidden_bias)

            current_layer = tf.nn.relu(
                tf.matmul(current_layer, hidden_weights) + hidden_bias)

        if isinstance(output_nodes, tuple):
            output_nodes = reduce(operator.mul, input_nodes, 1)

        # for some reason having output std divided by np.sqrt(output_nodes) massively outperforms np.sqrt(hidden_nodes)
        output_weights = tf.Variable(
            tf.truncated_normal((hidden_nodes, output_nodes), stddev=1. / np.sqrt(output_nodes)), name="output_weights")
        output_bias = tf.Variable(tf.constant(0.01, shape=(output_nodes,)), name="output_bias")

        variables.append(output_weights)
        variables.append(output_bias)

        output_layer = tf.matmul(current_layer, output_weights) + output_bias
        if output_softmax:
            output_layer = tf.nn.softmax(output_layer)

    return input_layer, output_layer, variables


def save_network(session, tf_variables, file_path):
    """Save the given set of variables to the given file using the given session

    Args:
        session (tf.Session): session within which the variables has been initialised
        tf_variables (list of tf.Variable): list of variables which will be saved to the file
        file_path (str): path of the file we want to save to.
    """
    variable_values = session.run(tf_variables)
    with open(file_path, mode='wb') as f:
        pickle.dump(variable_values, f)


def load_network(session, tf_variables, file_path):
    """Load the given set of variables from the given file using the given session

    Args:
        session (tf.Session): session within which the variables has been initialised
        tf_variables (list of tf.Variable): list of variables which will set up with the values saved to the file. List
            order matters, in must be the exact same order as was used to save and all of the same shape.
        file_path (str): path of the file we want to load from.
    """
    with open(file_path, mode='rb') as f:
        variable_values = pickle.load(f)

    try:
        if len(variable_values) != len(tf_variables):
            raise ValueError("Network in file had different structure, variables in file: %s variables in memeory: %s"
                             % (len(variable_values), len(tf_variables)))
        for value, tf_variable in zip(variable_values, tf_variables):
            session.run(tf_variable.assign(value))
    except ValueError as ex:
        # TODO: maybe raise custom exception
        raise ValueError("""Tried to load network file %s with different architecture from the in memory network.
Error was %s
Either delete the network file to train a new network from scratch or change the in memory network to match that dimensions of the one in the file""" % (file_path, ex))


def invert_board_state(board_state):
    """Returns the board state inverted, so all 1 are replaced with -1 and visa-versa

    Args:
        board_state (tuple of tuple of ints): The board we want to invert

    Returns:
        (tuple of tuple of ints) The board state for the other player
    """
    return tuple(tuple(-board_state[j][i] for i in range(len(board_state[0]))) for j in range(len(board_state)))


def get_stochastic_network_move(session, input_layer, output_layer, board_state, side,
                                valid_only=False, game_spec=None):
    """Choose a move for the given board_state using a stocastic policy. A move is selected using the values from the
     output_layer as a categorical probability distribution to select a single move

    Args:
        session (tf.Session): Session used to run this network
        input_layer (tf.Placeholder): Placeholder to the network used to feed in the board_state
        output_layer (tf.Tensor): Tensor that will output the probabilities of the moves, we expect this to be of
            dimesensions (None, board_squares) and the sum of values across the board_squares to be 1.
        board_state: The board_state we want to get the move for.
        side: The side that is making the move.

    Returns:
        (np.array) It's shape is (board_squares), and it is a 1 hot encoding for the move the network has chosen.
    """
    np_board_state = np.array(board_state)
    if side == -1:
        np_board_state = -np_board_state

    np_board_state = np_board_state.reshape(1, *input_layer.get_shape().as_list()[1:])
    probability_of_actions = session.run(output_layer,
                                         feed_dict={input_layer: np_board_state})[0]

    if valid_only:
        available_moves = list(game_spec.available_moves(board_state))
        if len(available_moves) == 1:
            move = np.zeros(game_spec.board_squares())
            np.put(move, game_spec.tuple_move_to_flat(available_moves[0]), 1)
            return move
        available_moves_flat = [game_spec.tuple_move_to_flat(x) for x in available_moves]
        for i in range(game_spec.board_squares()):
            if i not in available_moves_flat:
                probability_of_actions[i] = 0.

        prob_mag = sum(probability_of_actions)
        if prob_mag != 0.:
            probability_of_actions /= sum(probability_of_actions)

    try:
        move = np.random.multinomial(1, probability_of_actions)
    except ValueError:
        # sometimes because of rounding errors we end up with probability_of_actions summing to greater than 1.
        # so need to reduce slightly to be a valid value
        move = np.random.multinomial(1, probability_of_actions / (1. + 1e-6))

    return move


def get_deterministic_network_move(session, input_layer, output_layer, board_state, side, valid_only=False,
                                   game_spec=None):
    """Choose a move for the given board_state using a deterministic policy. A move is selected using the values from
    the output_layer and selecting the move with the highest score.

    Args:
        session (tf.Session): Session used to run this network
        input_layer (tf.Placeholder): Placeholder to the network used to feed in the board_state
        output_layer (tf.Tensor): Tensor that will output the probabilities of the moves, we expect this to be of
            dimesensions (None, board_squares).
        board_state: The board_state we want to get the move for.
        side: The side that is making the move.

    Returns:
        (np.array) It's shape is (board_squares), and it is a 1 hot encoding for the move the network has chosen.
    """
    np_board_state = np.array(board_state)
    np_board_state = np_board_state.reshape(1, *input_layer.get_shape().as_list()[1:])
    if side == -1:
        np_board_state = -np_board_state

    probability_of_actions = session.run(output_layer,
                                         feed_dict={input_layer: np_board_state})[0]

    if valid_only:
        available_moves = game_spec.available_moves(board_state)
        available_moves_flat = [game_spec.tuple_move_to_flat(x) for x in available_moves]
        for i in range(game_spec.board_squares()):
            if i not in available_moves_flat:
                probability_of_actions[i] = 0

    move = np.argmax(probability_of_actions)
    one_hot = np.zeros(len(probability_of_actions))
    one_hot[move] = 1.
    return one_hot
