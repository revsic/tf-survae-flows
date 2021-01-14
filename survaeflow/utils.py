import numpy as np
import tensorflow as tf


def gll(samples, mean, log_stddev=0.):
    """Compute gaussian log-likelihood.
    Args:
        samples: tf.Tensor, [tf.float32; [B, ...]], sample.
        mean: tf.Tensor, [tf.float32; [B, ...]], mean.
        log_stddev: tf.Tensor, [tf.float32; [B, ...]], log standard deviation.
    Returns:
        tf.Tensor, [tf.float32; [B]], gaussian log-likelihood.
    """
    return -0.5 * tf.reduce_sum(
        np.log(2 * np.pi) + log_stddev + \
            tf.exp(-2 * log_stddev) * tf.square(samples - mean),
        axis=tf.shape(samples)[1:])
