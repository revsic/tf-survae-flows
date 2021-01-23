import tensorflow as tf

from . import Distribution


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
        return tf.reduce_sum(
            -tf.nn.sigmoid_cross_entropy_with_logits(samples, logits=self.logits),
            axis=tf.shape(self.logits)[1:])

    def sample(self):
        """Sample from normal distribution (not backpropagatble).
        Returns:
            tf.Tensor, [tf.float32; [B, ...]], sampled.
        """
        prob = tf.sigmoid(self.logits)
        return tf.cast(
            tf.random.uniform(tf.shape(prob)) < prob, tf.float32)
