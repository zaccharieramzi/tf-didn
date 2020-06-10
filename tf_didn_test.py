import tensorflow as tf

from tf_didn import DIDN

def test_didn():
    n_filters = 8
    model = DIDN(
        n_filters=n_filters,
        n_dubs=2,
        n_filters_dub=n_filters,
        n_filters_recon=n_filters,
        n_convs_recon=3,
        n_scales=3,
    )
    model(tf.zeros((1, 64, 64, 1)))
