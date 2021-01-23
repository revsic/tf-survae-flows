import tensorflow as tf

from .. import Transform


class Scale(Transform):
    """Constant scaler.
    """
    def __init__(self, scale):
        """Initializer.
        Args:
            scale: tf.Tensor, [tf.float32; [B, ...]], scale factor,
                broadcastable to the inputs.
        """
        super(Scale, self).__init__()
        self.scale = scale
    
    def call(self, inputs):
        """Scale inputs.
        Args:
            inputs: tf.Tensor, [tf.flaot32; [B, ...]], inputs.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ...]], scaled.
            ldj: tf.Tensor, [tf.float32; []], log-determinant of jacobian.
        """
        # [B, ...]
        z = inputs * self.scale
        # []
        ldj = tf.reduce_sum(tf.math.log(tf.abs(self.scale)))
        return z, ldj
    
    def forward(self, inputs):
        """Scale inputs.
        Args:
            inputs: tf.Tensor, [tf.flaot32; [B, ...]], inputs.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ...]], scaled.
        """
        return inputs * self.scale
    
    def inverse(self, inputs):
        """Descale inputs.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], latent.
        Returns:
            tf.Tensor, [tf.float32; [B, ...]], recovered.
        """
        return inputs / self.scale
