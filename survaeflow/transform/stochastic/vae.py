from .. import Transform
from ...distribution.normal import Normal


class VAE(Transform):
    """Variational Autoencoder.
    """
    def __init__(self, encoder, decoder):
        """Initializer.
        Args:
            encoder: tf.keras.Model, latent encoder.
            decoder: tf.keras.Model, sampler.
        """
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def call(self, inputs):
        """Encode latent and compute likelihood contribution.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], input tensor.
        Returns:
            z: tf.Tensor, [tf.float32; [B, ...]], latent.
            ldj: tf.Tensor, [tf.float32; [B, ...]], likelihood contribution.
        """
        # [B, ...], [B, ...]
        mu, logstd = self.encoder(inputs)
        # [B, ...]
        posterior = Normal(mu, logstd)
        z = posterior.sample()
        # [B, ...]
        sample = self.decoder(z)
        # [B]
        ldj = Normal(inputs).log_prob(sample) - posterior.log_prob(z)
        return z, ldj

    def forward(self, inputs):
        """Encoder latent.
        Args:
            inputs: tf.Tensor, [tf.float32; [B, ...]], input tensor.
        Returns:
            z: tf.Tensor, [tf.flaot32; [B, ...]], latent.
        """
        # [B, ...], [B, ...]
        mu, logstd = self.encoder(inputs)
        # [B, ...]
        return Normal(mu, logstd).sample()
    
    def inverse(self, latent):
        """Sample from latent.
        Args:
            latent: tf.Tensor, [tf.float32; [B, ...]], latent tensor.
        Returns:
            tf.Tensor, [tf.float32; [B, ...]], sample.
        """
        return self.decoder(latent)
