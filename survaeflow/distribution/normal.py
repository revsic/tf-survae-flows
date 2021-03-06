import numpy as np
import tensorflow as tf

from .basedist import Distribution
from ..utils import sum_except_batch


class Normal(Distribution):
    """Gaussian normal distribution.
    """
    def __init__(self, mean, logstd=0.):
        """Initializer.
        Args:
            mean: tf.Tensor, [tf.float32; [B, ...]], mean tensor.
            logstd: tf.Tensor, [tf.float32; [B, ...]], log standard deviation.
        """
        super(Normal, self).__init__()
        self.mean = mean
        self.logstd = logstd

    def log_prob(self, samples):
        """Compute log-likelihood from samples.
        Args:
            samples: tf.Tensor, [tf.float32; [B, ...]], sample.
        Returns:
            tf.Tensor, [tf.float32; [B]], log-likelihood.
        """
        return -0.5 * sum_except_batch(
            np.log(2 * np.pi) + self.logstd + \
                tf.exp(-2 * self.logstd) * tf.square(samples - self.mean))
    
    def sample(self, shape=None):
        """Sample from normal distribution.
        Args:
            shape: Optional[Tuple[int]], sample size.
                If None, it will be replaced with shape of self.mean.
        Returns:
            tf.Tensor, [tf.float32; [B, ...]], sampled.
        """
        if shape is None:
            shape = tf.shape(self.mean)
        return tf.random.normal(shape) * tf.exp(self.logstd) + self.mean
