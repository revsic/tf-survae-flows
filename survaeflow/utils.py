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
