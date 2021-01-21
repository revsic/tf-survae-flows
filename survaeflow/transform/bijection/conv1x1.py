import tensorflow as tf

from .. import Transform


class Conv1x1(Transform):
    """1x1 invertible convolution.
    """
    def __init__(self):
        """Initializer.
        """
        super(Conv1x1, self).__init__()
    
    def build(self, input_shape):
        """Create parameters with given input shape.
        Args:
            input_shape: Tuple[int], input shape.
        """
        channels = tf.shape(input_shape)[-1]
        # [C, C]
        kernel = tf.random.normal([channels, channels])
        # [C, C], orthogonal initialization
        kernel, _ = tf.linalg.qr(kernel)
        self.kernel = tf.Variable(kernel)

    def call(self, inputs):
        """Run 1x1 convolution and compute log-determinant.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ..., C]], input tensor.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ..., C]], convolved tensor.
            ldj: tf.Tensor, [tf.float32; []], log-determinant of jacobian.
        """
        # [B, ..., C]
        z = tf.matmul(inputs, self.kernel)
        # []
        _, ldj = tf.linalg.slogdet(self.kernel)
        ldj = ldj * tf.reduce_prod(tf.shape(inputs)[1:-1])
        return z, ldj

    def forward(self, inputs):
        """Run 1x1 convolution.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ..., C]], input tensor.
        Returns:
            tf.Tensor, [tf.float32; [B, ..., C]], convolved tensor.
        """
        # since 1x1 convolution equals to fully-connected layer.
        return tf.matmul(inputs, self.kernel)

    def inverse(self, inputs):
        """Invert the 1x1 convolution.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ..., C]], latent.
        Returns:
            tf.Tensor, [tf.float32; [B, ..., C]], inverted.
        """
        return tf.matmul(inputs, tf.linalg.inv(self.kernel))
