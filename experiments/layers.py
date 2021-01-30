import tensorflow as tf

import survaeflow.transform.bijection as bij
from survaeflow import Flow
from survaeflow.distribution import Bernoulli, Normal
from survaeflow.transform import Transform
from survaeflow.utils import sum_except_batch


class Coupler(tf.keras.Model):
    """Affine coupler.
    """
    def __init__(self, model):
        """Initializer.
        Args:
            model: tf.keras.Model, compute affine parameters.
                inputs: [tf.float32; [B, ..., C]], latent.
                outputs: [tf.float32; [B, ..., C]], half for mean, half for logstd.
        """
        super().__init__()
        self.model = model

    def call(self, inputs):
        """Compute affine parameters, mean and logstd.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ..., C]], input tensor.
        Returns:
            mu: tf.Tensor, [tf.float32; [B, ..., C//2]], mean value.
            logstd: tf.Tensor, [tf.float32; [B, ..., C//2]], log-standard deviation.
        """
        # [B, C]
        x = self.model(inputs)
        # [B, C//2], [B, C//2]
        return tf.split(x, 2, axis=-1)


class FlowBaseline(Flow):
    """Invertible flow baseline.
    """
    def __init__(self, coupler, num_layers=4):
        """Initializer.
        Args:
            coupler: tf.keras.Model, affine coupler, compute on last axis.
                inputs: [tf.float32; [B, ..., C]], latent.
                outputs: [tf.float32; [B, ..., C]], half for mean, half for logstd.
        """
        networks = []
        for _ in range(num_layers):
            networks.extend([
                bij.AffineCoupling(Coupler(coupler)),
                bij.ActNorm(axis=-1),
                bij.Conv1x1()])
        super().__init__(Normal(0.), networks)


class ElemAbs(Transform):
    """Absolute first element of last axis.
    """
    def __init__(self, decoder):
        """Initializer.
        Args:
            decoder: tf.keras.Model, sign decoder.
                inputs: [tf.float32; [B, ..., C]], latent inputs.
                outputs: [tf.float32; [B, ..., 1]], bernoulli logits.
        """
        super().__init__()
        self.decoder = decoder

    def call(self, inputs):
        """Absolute first element and compute log-determinant of jacobian.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ..., C]], input tensor.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ..., C]], first-element absoluted.
            ldj: tf.Tensor, [tf.float32; [B]], log-likelihood contribution.
        """
        # [B, ..., 1]
        s = (tf.sign(inputs[..., 0:1]) + 1.)/ 2.
        # [B, ..., C]
        z = tf.concat([tf.abs(inputs[..., 0:1]), inputs[..., 1:]], axis=-1)
        # [B, ..., 1]
        logit = self.decoder(z)
        # [B]
        ldj = sum_except_batch(
            -tf.nn.sigmoid_cross_entropy_with_logits(s, logit))
        # [B, ..., C], [B]
        return z, ldj

    def forward(self, inputs):
        """Absolute first element.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ..., C]], input tensor.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ..., C]], first-element absoluted.
        """
        return tf.concat([tf.abs(inputs[..., 0:1]), inputs[..., 1:]], axis=-1)

    def inverse(self, z):
        """Recover sign of first element of last axis.
        Args:
            z: tf.Tensor, [tf.float32; [B, ..., C]], latent.
        Returns:
            tf.Tensor, [tf.flaot32; [B, ..., C]], sign recovered.
        """
        # [B, ..., 1]
        s = Bernoulli(self.decoder(z)).sample() * 2 - 1
        # [B, ..., C]
        return tf.concat([z[..., 0:1] * s, z[..., 1:]], axis=-1)

