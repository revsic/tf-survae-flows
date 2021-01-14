import tensorflow as tf


class Transform(tf.keras.Model):
    """Abstraction of tensor transform.
    """
    def __init__(self):
        """Initializer.
        """
        super(Transform, self).__init__()
    
    def call(self, *args, **kwargs):
        """Compute latent and log-likelihood contribution.
        """
        raise NotImplementedError('Transform.call is not implemented')

    def forward(self, *args, **kwargs):
        """Forward transform.
        """
        raise NotImplementedError('Transform.forward is not implemented')

    def inverse(self, *args, **kwargs):
        """Inverse transform.
        """
        raise NotImplementedError('Transform.inverse is not implemented')

