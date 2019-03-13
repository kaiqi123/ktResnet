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

    x = tf.zeros(shape=(2, 3), dtype=tf.float32)

    bn = tf.keras.layers.BatchNormalization(axis=-1, trainable=True)
    norm = bn(x, training=phase_train)
    y = tf.keras.layers.Dense(4)(norm)

    print('variables: %d' % len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
    print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    print('trainable variables: %d' % len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print('n_update_ops: %d' % len(update_ops))
    print(update_ops)

    print('n_update_ops(bn): %d' % len(bn.updates))
    print(bn.updates)

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