import tensorflow as tf

from .. import Transform
from ...distribution.normal import Normal


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
        ldj = Normal(x2).log_prob(sample)
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
        x2 = Normal(self.decoder(x1)).sample()
        # [B, ..., C, ...]
        return tf.concat([x1, x2], axis=self.axis)


class GenSlice(Transform):
    """Tensor slice for generative surjection.
    """
    def __init__(self, slice, decoder, axis=-1):
        """Initializer.
        Args:
            slice: int, slice size.
            decoder: tf.keras.Model, posterior approximator (deterministic).
            axis: int, slice axis.
        """
        super(GenSlice, self).__init__()
        self.slice = slice
        self.decoder = decoder
        self.axis = axis
    
    def call(self, inputs):
        """Recover the sliced tensor and compute log-determinant.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ..., slice, ...]], input tensor.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ..., C, ...]], recovered tensor.
            ldj: float, likelihood contribution.
        """
        z1 = inputs
        # [B, ..., C - slice, ...]
        posterior = Normal(self.decoder(inputs))
        # [B, ..., C - slice, ...]
        z2 = posterior.sample()
        # [B, ..., C, ...]
        z = tf.concat([z1, z2], axis=self.axis)
        # []
        ldj = -posterior.log_prob(z2)
        return z, ldj

    def forward(self, inputs):
        """Recover the removed tensor.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ..., slice, ...]], input tensor.
        Returns:
            tf.Tensor, [tf.float32; [B, ..., C, ...]], recovered tensor.
        """
        z1 = inputs
        # [B, ..., C - slice, ...]
        z2 = Normal(self.decoder(inputs)).sample()
        # [B, ..., C, ...]
        return tf.concat([z1, z2], axis=self.axis)

    def inverse(self, inputs):
        """Slice the inputs to transform to the latent.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ..., C, ...]], input tensor.
        Returns:
            tf.Tensor, [tf.float32; [B, ..., slice, ...]], sliced tensor.
        """
        # C
        channels = tf.shape(inputs)[self.axis]
        # [B, ..., slice, ...], [B, ..., C - slice, ...]
        x1, _ = tf.split(inputs, [self.slice, channels - self.slice], self.axis)
        # [B, ..., slice, ...]
        return x1
