import tensorflow as tf


class Checkerboard:
    """Checkerboard dataset.
    """
    def __init__(self, total, batch):
        """Initializer.
        Args:
            total: int, the number of the total points.
            batch: int, the size of the batch.
        """
        self.total = total
        self.batch = batch

    def __iter__(self):
        """Create sampler.
        Returns:
            Sampler, data point sampler.
        """
        return Checkerboard.Sampler(self.total, self.batch)

    class Sampler:
        """Data point sampler.
        """
        def __init__(self, total, batch):
            """Initializer.
            Args:
                total: int, the number of the total points.
                batch: int, the size of the batch.
            """
            self.total = total
            self.batch = batch
            self.last = 0
        
        def __next__(self):
            """Generate sample.
            """
            if self.last >= self.total:
                raise StopIteration()
            self.last += self.batch
            return self.sample(self.batch)

        @staticmethod
        def sample(size):
            """Sample data points, ref https://github.com/didriknielsen/survae_flows
            Args:
                size: int, the number of the points.
            Returns:
                np.ndarray, [np.int32; [size, 2]]
            """
            # [B], [-2, 2)
            x1 = tf.random.uniform([size], -2, 2)
            # [B]
            x2 = tf.random.uniform([size]) - \
                tf.cast(tf.random.uniform([size], 0, 2, dtype=tf.int32), tf.float32) * 2
            # [B]
            x2 = x2 + tf.math.floor(x1) % 2
            return tf.stack([x1, x2], axis=1) * 2
