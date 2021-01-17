import tensorflow as tf

from . import Transform


class Round(Transform):
    """Rounding inputs.
    """
    def __init__(self, decoder):
        """Initializer.
        Args:
            decoder: tf.keras.Model, posterior approximator, range [0, 1).
        """
        super(Round, self).__init__()
        self.decoder = decoder
    
    def call(self, inputs):
        """Dequantize inputs and compute log-determinant.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], input tensor.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ...]], dequantized.
            ldj: float, 0, log-determinant.
        """
        return inputs + self.decoder(inputs), 0.

    def forward(self, inputs):
        """Dequantize inputs.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], input tensor.
        Returns:
            tf.Tensor, [tf.float32; [B, ...]], dequantized.
        """
        return inputs + self.decoder(inputs)
    
    def inverse(self, inputs):
        """Round inputs.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], input tensor.
        Returns:
            tf.Tensor, [tf.float32; [B, ...]], rounded.
        """
        return tf.round(inputs)
