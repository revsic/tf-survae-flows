import tensorflow as tf

from . import Transform
from ..utils import gll


class Slice(Transform):
    """Tensor slice for inference surjection.
    """
    def __init__(self, slice, decoder, axis=-1):
        """Initializer.
        Args:
            slice: int, slice size.
            decoder: tf.keras.Model, posterior approximator.
            axis: int, slice axis.
        """
        super(Slice, self).__init__()
        self.slice = slice
        self.decoder = decoder
        self.axis = axis
    
    def call(self, inputs):
        """Slice the input tensor.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ..., C, ...]], input tensor.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ..., slice, ...]], sliced tensor.
            ldj: tf.Tensor, [tf.float32; [B]], log-likelihood contribution.
        """
        channels = tf.shape(inputs)[self.axis]
        # [B, ..., slice, ...], [B, ..., C - slice, ...]
        x1, x2 = tf.split(inputs, [self.slice, channels - self.slice], self.axis)
        # [B, ..., slice, ...]
        z = x1
        # [B, ..., C - slice, ...]
        sample = self.decoder(z)
        # [B]
        ldj = gll(sample, x2)
        # [B, ..., slice, ...], [B]
        return z, ldj

    def forward(self, inputs):
        """Slice the inputs to transform to the latent.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ..., C, ...]], input tensor.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ..., slice, ...]], sliced tensor.
        """
        # C
        channels = tf.shape(inputs)[self.axis]
        # [B, ..., slice, ...], [B, ..., C - slice, ...]
        x1, _ = tf.split(inputs, [self.slice, channels - self.slice], self.axis)
        # [B, ..., slice, ...]
        return x1

    def inverse(self, inputs):
        """Recover the removed tensor.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ..., slice, ...]], sliced tensor.
        Returns:
            x: tf.Tensor, [tf.float32; [B, ..., C, ...]], recovered tensor.
        """
        x1 = inputs
        # [B, ..., C - slice, ...]
        x2 = self.decoder(x1)
        # [B, ..., C, ...]
        return tf.concat([x1, x2], axis=self.axis)
