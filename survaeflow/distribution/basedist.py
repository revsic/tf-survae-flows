class Distribution:
    """Distribution baseline.
    """
    def __init__(self):
        """Initializer.
        """
        pass

    def log_prob(self, sample, *args, **kwargs):
        """Compute log-likelihood of given sample.
        """
        raise NotImplementedError('Distribution.log_prob is not implemented')

    def sample(self, *args, **kwargs):
        """Sample from distribution.
        """
        raise NotImplementedError('Distribution.sample is not implemented')
