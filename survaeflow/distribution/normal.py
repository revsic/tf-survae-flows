import numpy as np
import tensorflow as tf

from . import Distribution


class Normal(Distribution):
    """Gaussian normal distribution.
    """
    def __init__(self, mean, std):
        """Initializer.
        Args:
            mean: tf.Tensor, [tf.float32; [B, ...]], mean tensor.
            std: tf.Tensor, [tf.float32; [B, ...]], standard deviation.
        """
        super(Distribution, self).__init__()
        self.mean = mean
        self.std = std

    def log_prob(self, samples):
        """Compute log-likelihood from samples.
        Args:
            samples: tf.Tensor, [tf.float32; [B, ...]], sample.
        Returns:
            tf.Tensor, [tf.float32; [B]], log-likelihood.
        """
        return -0.5 * tf.reduce_sum(
            tf.log(2 * np.pi * self.std) + \
                self.std ** -2 * tf.square(samples - self.mean),
            axis=tf.shape(samples)[1:])
    
    def sample(self):
        """Sample from normal distribution.
        Returns:
            tf.Tensor, [tf.float32; [B, ...]], sampled.
        """
        return tf.random.normal(tf.shape(self.mean)) * self.std + self.mean
