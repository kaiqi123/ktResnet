import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.python.keras import backend as K


def test(phase_train):

    #K.set_learning_phase(1)

    if phase_train:
        K.clear_session()
        K.set_learning_phase(1)
    else:
        K.clear_session()
        K.set_learning_phase(0)


    x = tf.zeros(shape=(2, 3), dtype=tf.float32)

    bn = tf.keras.layers.BatchNormalization(axis=-1, trainable=True)
    normed = bn(x, training=phase_train)

    normed1 = bn(normed, training=phase_train)

    y = tf.keras.layers.Dense(4)(normed1)

    print('variables: %d' % len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
    print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    print('trainable variables: %d' % len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print('n_update_ops: %d' % len(update_ops))

    print('n_update_ops(bn): %d' % len(bn.updates))
    print(bn.updates)

    print("----------------------")

test(1)
test(0)