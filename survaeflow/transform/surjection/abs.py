import tensorflow as tf

from .. import Transform
from ...distribution.bernoulli import Bernoulli


class Abs(Transform):
    """Absolute operation.
    """
    def __init__(self, decoder):
        """Initializer.
        Args:
            decoder: tf.keras.Model, posterior approximator.
        """
        super(Abs, self).__init__()
        self.decoder = decoder
    
    def call(self, inputs):
        """Absolute inputs and compute log-determinant.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], input tensor.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ...]], latent value.
            ldj: tf.Tensor, [tf.float32; [B]], log-determinant.
        """ 
        # [B, ...]
        z = tf.abs(inputs)
        # [B, ...]
        sign = tf.sign(inputs)
        # [B], label negative as 0, positive as 1
        ldj = Bernoulli(self.decoder(inputs)).log_prob((sign + 1) / 2)
        return z, ldj

    def forward(self, inputs):
        """Absolute operation.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], input tensor.
        Returns:
            tf.Tensor, [tf.float32; [B, ...]], absolute value tensor.
        """
        return tf.abs(inputs)

    def inverse(self, inputs):
        """Recover the sign of the input tensor.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], absolute value tensor.
        Returns:
            tf.Tensor, [tf.float32; [B, ...]], sign recovered.
        """
        # [B, ...]
        logits = self.decoder(inputs)
        # [B, ...]
        sign = Bernoulli(logits).sample() * 2 - 1
        # [B, ...]
        return inputs * sign
