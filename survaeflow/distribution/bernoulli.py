import tensorflow as tf

from . import Distribution
from ..utils import sum_except_batch


class Bernoulli(Distribution):
    """Bernoulli distribution.
    """
    def __init__(self, logits):
        """Initializer.
        Args:
            logits: tf.Tensor, [tf.float32; [B, ...]], sigmoid logits tensor.
        """
        super(Bernoulli, self).__init__()
        self.logits = logits

    def log_prob(self, samples):
        """Compute log-likelihood from samples.
        Args:
            samples: tf.Tensor, [tf.float32; [B, ...]], sample.
        Returns:
            tf.Tensor, [tf.float32; [B]], log-likelihood.
        """
        return sum_except_batch(
            -tf.nn.sigmoid_cross_entropy_with_logits(samples, logits=self.logits))

    def sample(self, shape=None):
        """Sample from normal distribution (not backpropagatble).
        Args:
            shape: Optional[Tuple[int]], sample size.
                If None, it will be replaced with shape of self.logits.
        Returns:
            tf.Tensor, [tf.float32; [B, ...]], sampled.
        """
        prob = tf.sigmoid(self.logits)
        if shape is None:
            shape = tf.shape(prob)
        return tf.cast(
            tf.random.uniform(shape) < prob, tf.float32)
