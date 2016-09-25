import tensorflow as tf
import numpy as np
import pickle


def create_network(input_nodes, hidden_nodes, output_nodes=None, output_softmax=True):
    """Create a network with relu activations at each layer

    Args:
        output_nodes: (int): Number of output nodes, if None then number of input nodes is used
        input_nodes (int): The size of the board this network will work on. The output layer will also be this size
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

        output_weights = tf.Variable(
            tf.truncated_normal((hidden_nodes, output_nodes), stddev=1. / np.sqrt(hidden_nodes)), name="output_weights")
        output_bias = tf.Variable(tf.constant(0.01, shape=(output_nodes,)), name="output_bias")

        variables.append(output_weights)
        variables.append(output_bias)

        output_layer = tf.matmul(current_layer, output_weights) + output_bias
        if output_softmax:
            output_layer = tf.nn.softmax(output_layer)

    return input_layer, output_layer, variables


def save_network(session, variables, file_path):
    variable_values = session.run(variables)
    with open(file_path, mode='w') as f:
        pickle.dump(variable_values, f)


def load_network(session, tf_variables, file_path):
    with open(file_path, mode='r') as f:
        variable_values = pickle.load(f)
    for value, tf_variable in zip(variable_values, tf_variables):
        session.run(tf_variable.assign(value))


def invert_board_state(board_state):
    return tuple(tuple(-board_state(j, i) for i in range(len(board_state[0]))) for j in range(len(board_state)))


def get_stochastic_network_move(session, input_layer, output_layer, board_state, side):
    board_state_flat = np.ravel(board_state)
    if side == -1:
        board_state_flat = -board_state_flat

    probability_of_actions = session.run(output_layer,
                                         feed_dict={input_layer: [board_state_flat.ravel()]})[0]

    try:
        move = np.random.multinomial(1, probability_of_actions)
    except ValueError:
        # sometimes because of rounding errors we end up with probability_of_actions summing to greater than 1.
        # so need to reduce slightly to be a valid value
        move = np.random.multinomial(1, probability_of_actions / (sum(probability_of_actions) + 1e-7))

    return move


def get_deterministic_network_move(session, input_layer, output_layer, board_state, side):
    board_state_flat = np.ravel(board_state)
    if side == -1:
        board_state_flat = -board_state_flat

    probability_of_actions = session.run(output_layer,
                                         feed_dict={input_layer: [board_state_flat.ravel()]})[0]

    move = np.argmax(probability_of_actions)
    one_hot = np.zeros(len(probability_of_actions))
    one_hot[move] = 1.
    return one_hot
