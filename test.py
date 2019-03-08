import tensorflow as tf

init = tf.constant_initializer(value=0)
print(a)

print('fitting shape:')
with tf.Session():
    x = tf.get_variable('x', initializer=init)
    x.initializer.run()
    print(x.eval())
