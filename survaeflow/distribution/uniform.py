import tensorflow as tf

from . import Distribution


class Uniform(Distribution):
    """Uniform distribution.
    """
    def __init__(self, min=0., max=1.):
        """Initializer, assume the range [min, max).
        Args:
            min: tf.Tensor, [tf.float32; [B, ...]], min value.
            max: tf.Tensor, [tf.float32; [B, ...]], max value.
        """
        super(Uniform, self).__init__()
        self.min = min
        self.max = max

    def log_prob(self, samples):
        """Compute log-likelihood from samples.
        Args:
            samples: tf.Tensor, [tf.float32; [B, ...]], sample.
        Returns:
            tf.Tensor, [tf.float32; [B]], log-likelihood.
        """
        lb = tf.cast(self.min <= samples, tf.float32)
        ub = tf.cast(self.max > samples, tf.float32)
        return tf.reduce_sum(
            tf.math.log(lb * ub) - tf.math.log(self.max - self.min),
            axis=tf.shape(samples)[1:])

    def sample(self, shape=None):
        """Sample from normal distribution (not backpropagatble).
        Args:
            shape: Optional[Tuple[int]], sample size.
                If None, it will be replaced with shape of self.logits.
        Returns:
            tf.Tensor, [tf.float32; [B, ...]], sampled.
        """
        if shape is None:
            if isinstance(self.min, float):
                raise ValueError(
                    'if shape is None, self.mean should be tensor or numpy array')
            shape = tf.shape(self.min)
        return tf.random.uniform(shape, self.min, self.max)
