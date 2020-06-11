import numpy as np
import tensorflow as tf

from didn import DIDN

def test_didn():
    n_filters = 8
    model = DIDN(
        n_filters=n_filters,
        n_dubs=2,
        n_filters_recon=n_filters,
        n_convs_recon=3,
        n_scales=3,
    )
    model(tf.zeros((1, 64, 64, 1)))

def test_didn_change():
    n_filters = 8
    model = DIDN(
        n_filters=n_filters,
        n_dubs=2,
        n_filters_recon=n_filters,
        n_convs_recon=3,
        n_scales=3,
    )
    x = tf.random.normal((1, 64, 64, 1))
    y = x
    model(x)
    before = [v.numpy() for v in model.trainable_variables]
    model.compile(optimizer='sgd', loss='mse')
    model.train_on_batch(x, y)
    after = [v.numpy() for v in model.trainable_variables]
    for b, a in zip(before, after):
        assert np.any(np.not_equal(b, a))
