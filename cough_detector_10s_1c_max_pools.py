import tensorflow as tf
from global_weighted_pooling import global_weighted_pooling_layer


class CoughDetectorModel(object):

    def __init__(self, activation=tf.nn.relu):
        self.activation = activation

    def forward(self, input_tensor, is_train):
        self.network = input_tensor
        # print self.network.shape

        self.network = tf.layers.batch_normalization(self.network, training=is_train)
        self.network = tf.layers.conv2d(self.network, 64, (5, 5), strides=1, activation=self.activation, padding='SAME')
        self.network = tf.layers.max_pooling2d(self.network, (3, 1), strides=(2, 1))
        self.network = tf.layers.batch_normalization(self.network, training=is_train)
        self.network = tf.layers.conv2d(self.network, 64, (5, 3), strides=1, activation=self.activation, padding='SAME')
        self.network = tf.layers.max_pooling2d(self.network, (3, 1), strides=(2, 1))
        self.network = tf.layers.batch_normalization(self.network, training=is_train)
        self.network = tf.layers.conv2d(self.network, 64, (5, 3), strides=1, activation=self.activation, padding='SAME')
        self.network = tf.layers.max_pooling2d(self.network, (3, 1), strides=(2, 1))
        self.network = tf.layers.batch_normalization(self.network, training=is_train)
        self.network = tf.layers.conv2d(self.network, 64, (5, 3), strides=1, activation=self.activation, padding='SAME')
        self.network = tf.layers.max_pooling2d(self.network, (3, 1), strides=(2, 1))
        self.network = tf.layers.batch_normalization(self.network, training=is_train)
        self.network = tf.layers.conv2d(self.network, 64, (5, 3), strides=1, activation=self.activation, padding='SAME')
        self.network = tf.layers.max_pooling2d(self.network, (3, 1), strides=(2, 1))
        self.network = tf.layers.batch_normalization(self.network, training=is_train)
        # self.network = tf.layers.conv2d(self.network, 64, (5, 3), strides=1, activation=self.activation, padding='SAME')
        # self.network = tf.layers.max_pooling2d(self.network, (3, 1), strides=(2, 1))
        # self.network = tf.layers.batch_normalization(self.network, training=is_train)
        self.network = tf.layers.conv2d(self.network, 1, (1, 1), strides=1, activation=self.activation, padding='SAME')
        self.network = tf.layers.batch_normalization(self.network, training=is_train)

        self.network = tf.transpose(self.network, perm=[0, 1, 3, 2])
        self.network = tf.layers.dense(self.network, 2048)
        # self.network = tf.layers.dropout(self.network, rate=0.2)
        self.network = tf.layers.dense(self.network, 2048)
        self.network = tf.layers.dense(self.network, 311)
        self.network = tf.transpose(self.network, perm=[0, 1, 3, 2])

        self.network = tf.nn.sigmoid(self.network, name='output')

        return self.network
