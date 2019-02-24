import tensorflow as tf

class DataInput(object):

    def __init__(self):
        self.example_batch = None
        self.label_batch = None

    def input_data_into_pipeline(self, dataset_file, batch_size, image_width, image_height, num_channels, seed, pad, datasetName):

        filename_queue = tf.train.string_input_producer([dataset_file], num_epochs=None)
        reader = tf.TextLineReader()
        key_temp, value_temp = reader.read(filename_queue)
        record_defaults = [[1], ['']]
        col1, col2 = tf.decode_csv(value_temp, record_defaults=record_defaults)

        file_content = tf.read_file(col2)
        train_image = tf.image.decode_png(file_content, channels=num_channels)
        train_image = tf.image.per_image_standardization(train_image)
        train_image = tf.image.resize_image_with_pad(train_image, image_width + pad, image_width + pad)
        train_image = tf.image.random_flip_left_right(train_image)
        train_image = tf.random_crop(train_image, [image_height, image_width, 3], seed=seed, name="crop")

        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size
        self.example_batch, self.label_batch = tf.train.shuffle_batch (
                [train_image, col1], batch_size=batch_size, capacity=capacity,
                    min_after_dequeue=min_after_dequeue, seed=seed)

        return self.example_batch, self.label_batch
