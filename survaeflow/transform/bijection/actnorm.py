import tensorflow as tf

from .. import Transform


class ActNorm(Transform):
    """Activation normalization.
    """
    def __init__(self, axis=-1):
        """Initializer.
        Args:
            axis: int, target axis.
        """
        super(ActNorm, self).__init__()
        self.axis = axis
        self.init = tf.Variable(0., trainable=False)
    
    def build(self, input_shape):
        """Generate the parameters.
        Args:
            input_shape: Tuple[int], input shape.
        """
        self.mult = tf.Variable(1., trainable=False)
        self.mu = tf.Variable(
            tf.zeros(shape=(input_shape[self.axis],)))
        self.log_sigma = tf.Variable(
            tf.zeros(shape=(input_shape[self.axis],)))

    def initialized(self):
        """Check whether this object is initialized with ddi or not.
        Returns:
            bool, whether initialized or not.
        """
        return self.init == 1.

    def ddi(self, inputs):
        """Data-dependent initialization.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ..., C, ...]], input tensor.
        """
        dim = len(inputs.shape)
        target = self.axis
        if target < 0:
            target += dim
        
        axis = list(range(dim))
        axis = axis[:target] + axis[target + 1:]
        # [C], [C]
        mean, variance = tf.nn.moments(inputs, axes=axis)
        # assign input stats
        self.mu.assign(mean)
        self.log_sigma.assign(0.5 * tf.math.log(variance + 1e-5))
        # update multiplier
        self.mult.assign(
            tf.cast(tf.reduce_prod(tf.shape(inputs)[axis[1:]]), tf.float32))
        self.init.assign(1.)

    def call(self, inputs):
        """Run activation normalization and compute log-determinant of jacobian.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ..., C, ...]], input tensor.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ..., C, ...]], normalized.
            ldj: tf.Tensor, [tf.float32; []], log-determinant of jacobian.
        """
        if not self.initialized():
            self.ddi(inputs)
        # [B, ..., C, ...]        
        z = (inputs - self.mu) / tf.exp(self.log_sigma)
        # []
        ldj = tf.reduce_sum(-self.log_sigma) * self.mult
        return z, ldj

    def forward(self, inputs):
        """Run activation normalization.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ..., C, ...]], input tensor.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ..., C, ...]], latent vector.
        Raises:
            RuntimeError, if method `call` is not called before.
        """
        if not self.initialized():
            raise RuntimeError('method `call` is not called before')
        return (inputs - self.mu) / tf.exp(self.log_sigma)

    def inverse(self, inputs):
        """Run activation normalization.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ..., C, ...]], latent.
        Returns:
            tf.Tensor, [tf.float32; [B, ..., C, ...]], recovered tensor.
        Raises:
            RuntimeError, if method `call` is not called before.
        """
        if not self.initialized:
            raise RuntimeError('method `call` is not called before')
        return inputs * tf.exp(self.log_sigma) + self.mu
