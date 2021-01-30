from .. import Transform


class Shift(Transform):
    """Constant Shifter.
    """
    def __init__(self, shift):
        """Initializer.
        Args:
            shift: tf.Tensor, [tf.float32; [B, ...]], shift bias,
                broadcastable to the inputs.
        """
        super(Shift, self).__init__()
        self.shift = shift
    
    def call(self, inputs):
        """Shift inputs.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], inputs.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ...]], shifted.
            ldj: float, zero.
        """
        # [B, ...]
        z = inputs + self.shift
        return z, 0.
    
    def forward(self, inputs):
        """Shift inputs.
        Args:
            inputs: tf.Tensor, [tf.flaot32; [B, ...]], inputs.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ...]], shifted.
        """
        return inputs + self.shift
    
    def inverse(self, inputs):
        """Unshift inputs.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], latent.
        Returns:
            tf.Tensor, [tf.float32; [B, ...]], recovered.
        """
        return inputs - self.shift
