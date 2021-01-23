from numpy.lib.financial import irr
import tensorflow as tf

from .. import Transform


class Sigmoid(Transform):
    """Element-wise sigmoid.
    """
    def __init__(self, temperature=1., eps=0.):
        """Initializer.
        Args:
            temperature: float, input temperature (scaling factor).
            eps: float, small value for preventing numerical error (over, underflow).
        """
        super(Sigmoid, self).__init__()
        self.temperature = temperature
        self.eps = eps
    
    def call(self, inputs):
        """Compute sigmoid and log-determinant.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], input tensor.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ...]], latent.
            ldj: tf.Tensor, [tf.float32; [B]], log-determinant of jacobian.
        """
        # [B, ...]
        z = tf.nn.sigmoid(self.temperature * inputs)
        # [B]
        ldj = tf.reduce_sum(
            tf.math.log(self.temperature) - tf.nn.softplus(-inputs) - tf.nn.softplus(inputs),
            axis=tf.shape(inputs)[1:])
        return z, ldj

    def forward(self, inputs):
        """Compute sigmoid.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], input tensor.
        Returns:
            tf.Tensor, [tf.float32; [B, ...]], latent.
        """
        return tf.nn.sigmoid(self.temperature * inputs)

    def inverse(self, inputs):
        """Invert sigmoid.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], latent.
        Returns:
            tf.Tensor, [tf.float32; [B, ...]], inverted.
        """
        # [B, ...]
        z  = tf.clip_by_value(inputs, self.eps, 1 - self.eps)
        # [B, ...]
        return (tf.math.log(z) - tf.math.log(1 - z)) / self.temperature


class Logit(Transform):
    """Element-wise logits.
    """
    def __init__(self, temperature=1., eps=1e-6):
        """Initializer.
        Args:
            temperature: float, input temperature (scaling factor).
            eps: float, small value for preventing numerical error (over, underflow).
        """
        super(Logit, self).__init__()
        self.temperature = temperature
        self.eps = eps
    
    def call(self, inputs):
        """Compute logits and log-determinant.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], input tensor,
                should be in range [0, 1).
        Returns:
            z: tf.Tensor, [tf.float32; [B, ...]], latent.
            ldj: tf.Tensor, [tf.float32; [B]], log-determinant of jacobian.
        """
        # [B, ...]
        x = tf.clip_by_value(inputs, self.eps, 1 - self.eps)
        # [B, ...]
        ir = (tf.math.log(x) - tf.math.log(1 - x))
        # [B, ...]
        z = ir / self.temperature
        # [B]
        ldj = -tf.reduce_mean(
            tf.math.log(self.temperature) - tf.nn.softplus(-ir) - tf.nn.softplus(ir),
            axis=tf.shape(inputs)[1:])
        return z, ldj
    
    def forward(self, inputs):
        """Compute logits.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], input tensor,
                should be in range [0, 1).
        Returns:
            z: tf.Tensor, [tf.float32; [B, ...]], latent.
        """
        # [B, ...]
        x = tf.clip_by_value(inputs, self.eps, 1 - self.eps)
        # [B, ...]
        return (tf.math.log(x) - tf.math.log(1 - x)) / self.temperature
    
    def inverse(self, inputs):
        """Compute sigmoid.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], input tensor.
        Returns:
            tf.Tensor, [tf.float32 [B, ...]], inverted logits (sigmoid).
        """
        return tf.nn.sigmoid(inputs * self.temperature)
