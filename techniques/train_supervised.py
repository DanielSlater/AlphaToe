import os
import random

import tensorflow as tf

from common.network_helpers import save_network, load_network


def train_supervised(game_spec, create_network, network_file_path,
                     positions,
                     test_set_ratio=0.4,
                     regularization_coefficent=1e-5,
                     batch_size=100,
                     learn_rate=1e-4,
                     stop_turns_without_improvement = 7):
    """Train a network using supervised learning using against a list of game positions and moves chosen.
    We stop after we have had stop_turns_without_improvement without an improvement in the test error.
    The test set is used as a validation set as well, will possibly improve this in the future to have a seperate test
     and validation set.

    Args:
        stop_turns_without_improvement (int): we stop training after this many iterations without any improvement in
            the test error.
        regularization_coefficent (float): amount to multiply the l2 regularizer by in the loss function
        test_set_ratio (float): portion of the data to divide into the test set,
        positions ([(board_state, move)]): list of tuples of board states and the moves chosen in those board_states
        game_spec (games.base_game_spec.BaseGameSpec): The game we are playing
        create_network (->(input_layer : tf.placeholder, output_layer : tf.placeholder, variables : [tf.Variable])):
            Method that creates the network we will train.
        network_file_path (str): path to the file with weights we want to load for this network
        learn_rate (float):
        batch_size (int):

    Returns:
        episode_number, train_error, train_accuracy, new_test_error, test_accuracy
    """
    input_layer, output_layer, variables = create_network()

    test_set_count = int(len(positions) * test_set_ratio)
    train_set = positions[:-test_set_count]
    test_set = positions[-test_set_count:]

    actual_move_placeholder = tf.placeholder("float", (None, game_spec.outputs()))

    error = tf.reduce_sum(tf.square(actual_move_placeholder - output_layer))

    regularizer = None
    for var in variables:
        if regularizer is None:
            regularizer = tf.nn.l2_loss(var)
        else:
            regularizer += tf.nn.l2_loss(var)

    loss = error + regularizer * regularization_coefficent

    train_step = tf.train.RMSPropOptimizer(learn_rate).minimize(loss)

    correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(actual_move_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        if os.path.isfile(network_file_path):
            print("loading existing network")
            load_network(session, variables, network_file_path)

        episode_number = 1
        turns_without_test_improvement = 0

        best_test_error, test_accuracy = session.run([error, accuracy],
                                                     feed_dict={
                                                         input_layer: [x[0] for x in test_set],
                                                         actual_move_placeholder: [x[1] for x in test_set]})

        while True:
            random.shuffle(train_set)
            train_error = 0

            for start_index in range(0, len(train_set) - batch_size + 1, batch_size):
                mini_batch = train_set[start_index:start_index + batch_size]

                batch_error, _ = session.run([error, train_step],
                                             feed_dict={input_layer: [x[0] for x in mini_batch],
                                                        actual_move_placeholder: [x[1] for x in mini_batch]})
                train_error += batch_error

            new_test_error, test_accuracy = session.run([error, accuracy],
                                                        feed_dict={input_layer: [x[0] for x in test_set],
                                                                   actual_move_placeholder: [x[1] for x in test_set]})

            print("episode: %s train_error: %s test_error: %s test_acc: %s" %
                  (episode_number, train_error, new_test_error, test_accuracy))

            if new_test_error < best_test_error:
                best_test_error = new_test_error
                turns_without_test_improvement = 0
            else:
                turns_without_test_improvement += 1
                if turns_without_test_improvement > stop_turns_without_improvement:
                    train_accuracy = session.run([accuracy], feed_dict={input_layer: [x[0] for x in train_set],
                                                                        actual_move_placeholder: [x[1] for x in
                                                                                                  train_set]})

                    print("test error not improving for %s turns, ending training" % (stop_turns_without_improvement, ))
                    break

            episode_number += 1

        print("final episode: %s train_error: %s train acc: %s test_error: %s test_acc: %s" %
              (episode_number, train_error, train_accuracy, new_test_error, test_accuracy))

        save_network(session, variables, network_file_path)

    return episode_number, train_error, train_accuracy, new_test_error, test_accuracy