import tensorflow as tf

from .. import Transform


class AffineCoupling(Transform):
    """Affine coupling layer.
    """
    def __init__(self, encoder, axis=-1):
        """Initializer.
        Args:
            encoder: tf.keras.Model, affine parameter encoder.
            axis: int, target axis.
        """
        super(AffineCoupling, self).__init__()
        self.encoder = encoder
        self.axis = axis

    def call(self, inputs):
        """Affine coupling bijection and compute log-determinant.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], input tensor.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ...]], latent.
            ldj: tf.Tensor, [tf.float32; [B]], log-determinant of jacobian.
        """
        # [B, ..., C // 2, ...], [B, ..., C // 2, ...]
        x1, x2 = tf.split(inputs, 2, axis=self.axis)
        # [B, ..., C // 2, ...], [B, ..., C // 2, ...]
        mu, logstd = self.encoder(x2)
        # [B, ..., C // 2, ...]
        x1 = x1 * tf.exp(logstd) + mu
        # [B, ...]
        z = tf.concat([x1, x2], axis=self.axis)
        # [B]
        ldj = tf.reduce_sum(logstd, axis=tf.shape(logstd)[1:])
        return z, ldj

    def forward(self, inputs):
        """Affine coupling bijection.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], input tensor.
        Returns:
            tf.Tensor, [tf.float32; [B, ...]], latent.
        """
        # [B, ..., C // 2, ...], [B, ..., C // 2, ...]
        x1, x2 = tf.split(inputs, 2, axis=self.axis)
        # [B, ..., C // 2, ...], [B, ..., C // 2, ...]
        mu, logstd = self.encoder(x2)
        # [B, ..., C // 2, ...]
        x1 = x1 * tf.exp(logstd) + mu
        # [B, ...]
        return tf.concat([x1, x2], axis=self.axis)
    
    def inverse(self, inputs):
        """Inverting affine coupling.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], latent.
        Returns:
            tf.Tensor, [tf.float32; [B, ...]], recovered sample.
        """
        # [B, ..., C // 2, ...], [B, ..., C // 2, ...]
        z1, z2 = tf.split(inputs, 2, axis=self.axis)
        # [B, ..., C // 2, ...]
        x2 = z2
        # [B, ..., C // 2, ...]
        mu, logstd = self.encoder(x2)
        # [B, ..., C // 2, ...]
        x1 = (z1 - mu) / tf.exp(logstd)
        # [B, ..., C // 2, ...]
        return tf.concat([x1, x2], axis=self.axis)
