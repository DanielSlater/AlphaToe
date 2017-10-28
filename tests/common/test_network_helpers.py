import os
from unittest import TestCase

import numpy as np
import tensorflow as tf

from common.network_helpers import create_network, save_network, load_network


class TestNetworkHelpers(TestCase):
    def test_create_network(self):
        input_nodes = 20
        hidden_nodes = (50, 40, 30)
        input_layer, output_layer, variables = create_network(input_nodes, hidden_nodes)
        self.assertSequenceEqual(input_layer.get_shape().as_list(), [None, input_nodes])
        self.assertSequenceEqual(output_layer.get_shape().as_list(), [None, input_nodes])
        self.assertEqual(len(variables), (len(hidden_nodes) + 1) * 2)

    def test_create_network_with_2d_input(self):
        input_nodes = (5, 5)
        hidden_nodes = (50, 40, 30)
        input_layer, output_layer, variables = create_network(input_nodes, hidden_nodes)
        self.assertSequenceEqual(input_layer.get_shape().as_list(), [None, input_nodes[0], input_nodes[1]])
        self.assertSequenceEqual(output_layer.get_shape().as_list(), [None, input_nodes[0] * input_nodes[1]])
        self.assertEqual(len(variables), (len(hidden_nodes) + 1) * 2)

    def test_save_and_load_network(self):
        try:
            file_name = 'test.p'
            input_nodes = 20
            hidden_nodes = (50, 40, 30)
            _, _, variables1 = create_network(input_nodes, hidden_nodes)
            _, _, variables2 = create_network(input_nodes, hidden_nodes)

            with tf.Session() as session:
                session.run(tf.global_variables_initializer())

                save_network(session, variables1, file_name)
                load_network(session, variables2, file_name)

                for var1, var2 in zip(variables1, variables2):
                    np.testing.assert_array_almost_equal(session.run(var1), session.run(var2))
        finally:
            try:
                os.remove(file_name)
            except OSError:
                pass

    def test_load_variables_into_network_of_wrong_size_gives_friendly_exception(self):
        try:
            file_name = 'test.p'
            input_nodes = 20

            _, _, variables1 = create_network(input_nodes, (30, ))
            _, _, variables2 = create_network(input_nodes, (40, ))

            with tf.Session() as session:
                session.run(tf.global_variables_initializer())

                save_network(session, variables1, file_name)

                with self.assertRaises(ValueError):
                    load_network(session, variables2, file_name)
        finally:
            try:
                os.remove(file_name)
            except OSError:
                pass