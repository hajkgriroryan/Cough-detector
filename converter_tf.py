import tensorflow as tf


def rgb_to_ycbcr(image):
    xform = tf.constant([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = tf.tensordot(image, tf.transpose(xform), 1)
    ycbcr += tf.constant([0, 128., 128.])

    return ycbcr


def ycbcr_to_rgb(image):
    xform2 = tf.constant([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = tf.cast(image, tf.float32)
    rgb -= tf.constant([0, 128., 128.])
    rgb = tf.tensordot(rgb, tf.transpose(xform2), 1)
    rgb = tf.clip_by_value(rgb, 0.0, 255.)

    return rgb
