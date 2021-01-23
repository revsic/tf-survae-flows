import tensorflow as tf


class Flow(tf.keras.Model):
    """Invertible flow.
    """
    def __init__(self, prior, networks):
        """Initializer.
        Args:
            prior: Distribution, latent prior.
            networks: List[tf.keras.Model], sequence of transforms.
        """
        super(Flow, self).__init__()
        self.prior = prior
        self.networks = networks
    
    def call(self, inputs):
        """Generate latent and compute log-determinant.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], input tensor.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ...]], latent tensor.
            ldj: tf.Tensor, [tf.float32; [B]], log-determinant of jacobian
                (log-likelihood contribution term).
        """
        z, ldj = inputs, 0.
        for net in self.networks:
            z, contrib = net(z)
            ldj = ldj + contrib
        return z, self.prior.log_prob(z) + ldj

    def forward(self, inputs):
        """Generate latent.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], input tensor.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ...]], latent tensor.
        """
        z = inputs
        for net in self.networks:
            z = net.forward(z)
        return z

    def inverse(self, latent):
        """Generate sample from latent.
        Args:
            latent: tf.Tensor, [tf.float32; [B, ...]], latent tensor.
        Returns:
            sample: tf.Tensor, [tf.float32; [B, ...]], sample tensor.
        """
        sample = latent
        for net in self.networks[::-1]:
            sample = net.inverse(sample)
        return sample

    def sample(self, shape):
        """Sample from prior.
        Args:
            shape: Tuple[int], shape of the prior.
        Returns:
            tf.Tensor, [tf.float32; [size, ...]], sampled.
        """
        latent = self.prior.sample(shape)
        return self.inverse(latent)
