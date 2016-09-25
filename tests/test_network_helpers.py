import os
from unittest import TestCase
import tensorflow as tf
import numpy as np
from network_helpers import create_network, save_network, load_network


class TestNetworkHelpers(TestCase):
    def test_create_network(self):
        input_nodes = 20
        hidden_nodes = (50, 40, 30)
        input_layer, output_layer, variables = create_network(input_nodes, hidden_nodes)
        self.assertSequenceEqual(input_layer.get_shape().as_list(), [None, input_nodes])
        self.assertSequenceEqual(output_layer.get_shape().as_list(), [None, input_nodes])
        self.assertEqual(len(variables), (len(hidden_nodes) + 1) * 2)

    def test_save_and_load_network(self):
        try:
            file_name = 'test.p'
            input_nodes = 20
            hidden_nodes = (50, 40, 30)
            _, _, variables1 = create_network(input_nodes, hidden_nodes)
            _, _, variables2 = create_network(input_nodes, hidden_nodes)

            with tf.Session() as session:
                session.run(tf.initialize_all_variables())

                save_network(session, variables1, file_name)
                load_network(session, variables2, file_name)

                for var1, var2 in zip(variables1, variables2):
                    np.testing.assert_array_almost_equal(session.run(var1), session.run(var2))
        finally:
            try:
                os.remove(file_name)
            except OSError:
                pass
