import io

import matplotlib.pyplot as plt
import tensorflow as tf


def sum_except_batch(inputs):
    """Reduce sum except batch.
    Args:
        inputs: tf.Tensor, [_; [B, ...]], input tensor.
    Returns:
        tf.Tensor, [_; [B]], reduced.
    """
    dim = len(inputs.shape)
    axis = list(range(dim))[1:]
    return tf.reduce_sum(inputs, axis=axis)


def mean_except_batch(inputs):
    """Reduce mean except batch.
    Args:
        inputs: tf.Tensor, [_; [B, ...]], input tensor.
    Returns:
        tf.Tensor, [_; [B]], reduced.
    """
    dim = len(inputs.shape)
    axis = list(range(dim))[1:]
    return tf.reduce_mean(inputs, axis=axis)


def pyplot_as_img():
    """Convert pyplot figure as image tensor.
    Returns:
        tf.Tensor, [_; [B, H, W, C]], image tensor.
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return tf.image.decode_png(buf.getvalue(), channels=3)
