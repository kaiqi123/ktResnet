import tensorflow as tf

def test():
    graph = tf.get_default_graph()
    tf.keras.backend.set_learning_phase(True)
    features = tf.zeros(shape=(3, 10), dtype=tf.float32)
    #normed = tf.keras.layers.BatchNormalization()(features)

    print('variables: %d' % len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print('n_ops:        %d' % len(graph.get_operations()))
    print('n_update_ops: %d' % len(update_ops))


def test2():
    tf.keras.backend.set_learning_phase(True)
    x = tf.zeros(shape=(2, 3), dtype=tf.float32)
    normed = tf.keras.layers.BatchNormalization()(x)
    y = tf.keras.layers.Dense(4)(normed)

    print('variables: %d' % len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
    print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print('n_update_ops: %d' % len(update_ops))


test2()