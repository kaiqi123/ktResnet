import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.python.keras import backend as K


def test(phase_train):

    #K.set_learning_phase(1)

    #if phase_train:
    #    K.clear_session()
    #   K.set_learning_phase(1)
    #else:
    #    K.clear_session()
    #    K.set_learning_phase(0)

    #K.clear_session()
    #K.set_learning_phase(phase_train)

    def batch_norm(x, phase_train):

        with tf.name_scope('bn') as scope:

            weight = tf.random_normal_initializer(mean=1, stddev=0.045)
            bias = tf.constant_initializer(value=0)
            moving_mean = tf.constant_initializer(value=0)
            moving_variance = tf.ones_initializer()

            #bn = tf.keras.layers.BatchNormalization(axis=-1, name='BatchNorm', trainable=self.trainable)

            """
            bn = tf.layers.batch_normalization(axis=-1, name='BatchNorm', trainable=True,
                                           beta_initializer=bias,
                                           gamma_initializer=weight,
                                           moving_mean_initializer=moving_mean,
                                           moving_variance_initializer=moving_variance)
            """

            bn = tf.contrib.layers.batch_norm(x, trainable=True, is_training=phase_train,
                                           updates_collections=tf.GraphKeys.UPDATE_OPS)

            print(weight)
            print(bias)
            print(moving_mean)
            print(moving_variance)
        return bn

    """
    x = tf.zeros(shape=(2, 3), dtype=tf.float32)
    #bn = tf.keras.layers.BatchNormalization(axis=-1, trainable=True)
    y1 = tf.keras.layers.Dense(6)(x)
    bn1 = batch_norm()
    norm = bn1(y1, training=phase_train)
    y = tf.keras.layers.Dense(4)(norm)
    bn2 = batch_norm()
    norm2 = bn2(y, training=phase_train)
    """

    x = tf.zeros(shape=(2, 3), dtype=tf.float32)
    y1 = tf.keras.layers.Dense(6)(x)
    norm = batch_norm(y1, phase_train)
    y = tf.keras.layers.Dense(4)(norm)
    norm2 = batch_norm(y, phase_train)

    print('variables: %d' % len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
    for e in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        print(e)

    print('trainable variables: %d' % len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    for e in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print(e)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print('n_update_ops: %d' % len(update_ops))
    #print(update_ops)
    for elem in update_ops:
        print(elem.name)

    #i2t_update_extra_ops = [elem for elem in update_ops if 'text_feature/attention' not in elem.name]
    #print(i2t_update_extra_ops)

    """
    update_ops1 = tf.get_collection(bn1.updates)
    print('n_update_ops(bn): %d' % len(update_ops1))
    print(update_ops1)
    #for e in bn.updates:
    #    print(e)


    update_ops2 = tf.get_collection(bn2.updates)
    print('n_update_ops(bn): %d' % len(update_ops1))
    print(update_ops2)
    #for e in bn.updates:
    #    print(e)

    update_ops3 = [tf.get_collection(bn1.updates),
                    tf.get_collection(bn2.updates)]

    print('n_update_ops(bn): %d' % len(update_ops3))
    print(update_ops3)
    #for e in bn.updates:
    #    print(e)
    """

    print("----------------------")

test(tf.constant(True))
test(tf.constant(False))

#test(1)
#test(0)

"""
a = tf.placeholder(tf.bool)  #placeholder for a single boolean value

# mode = None
def fc1():
    mode  = 1
    return mode
def fc2():
    mode  = 0
    return mode

b = tf.cond(tf.equal(a, tf.constant(True)), fc1, fc2)
c = tf.cast(b, tf.int32)
print(c)
#c = tf.Print(b)
#print(c.eval())
sess = tf.InteractiveSession()
res = sess.run(b, feed_dict = {a: True})
sess.close()
#print(mode)
"""