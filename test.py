import tensorflow as tf
import numpy as np

"""
v = np.array([1,2,3,4,5,6])
b = v.reshape((-1, 1))
print(v)
print(v.shape)
print(b)
print(b.shape)
"""

a = tf.constant_initializer(value=0)

with tf.Session():
    #x = tf.get_variable('x', shape=[2, 4], initializer=a)
    x = tf.get_variable('x', initializer=a)
    x.initializer.run()
    print(x.eval())


