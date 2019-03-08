import tensorflow as tf

value = [0, 1, 2, 3, 4, 5, 6, 7]
init = tf.constant_initializer(value=0)
print(init)


print('fitting shape:')
with tf.Session():
    x = tf.get_variable('x', shape=[2, 4], initializer=init)
    x.initializer.run()
    print(x.eval())


