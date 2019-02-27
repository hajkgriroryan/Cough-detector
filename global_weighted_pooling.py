import tensorflow as tf
import numpy as np


def get_weights(alpha, arr_size):
    weights = np.zeros((1, arr_size))
    for i in range(arr_size):
        weights[0, i] = alpha**i
    return weights


def global_weighted_pooling(input_tensor, weights):

    tensor = tf.transpose(input_tensor, (1, 0))
    sorted_tensor_by_columns = tf.nn.top_k(tensor, k=tensor.get_shape()[-1], sorted=True)[0]
    tensor = tf.transpose(sorted_tensor_by_columns)

    gwrp = tf.matmul(weights, tensor)  # gwrp = global weighted rank pooling

    return gwrp


def global_weighted_pooling_layer(inputs, alpha=0.7):
    assert inputs.get_shape()[3] == 1
    assert inputs.get_shape()[1] is not None

    weights = tf.constant(get_weights(alpha, inputs.get_shape()[1]), dtype=tf.float32)
    # do global weighted rank pooling for batch
    gwrp = tf.map_fn(lambda tensor: global_weighted_pooling(tensor[:, :, 0], weights), inputs)
    gwrp = tf.expand_dims(gwrp, axis=-1)

    return gwrp
