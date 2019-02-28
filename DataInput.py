import tensorflow as tf


class DataInput(object):

    def __init__(self, dataset_file, batch_size, image_width, image_height, num_channels, seed, pad, datasetName, whetherTrain):

        self.dataset_file = dataset_file
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.num_channels = num_channels
        self.seed = seed
        self.pad = pad
        self.datasetName = datasetName
        self.example_batch = None
        self.label_batch = None
        self.whetherTrain = whetherTrain

        self.input_data_into_pipeline()

    def input_data_into_pipeline(self):

        filename_queue = tf.train.string_input_producer([self.dataset_file], num_epochs=None)
        reader = tf.TextLineReader()
        key_temp, value_temp = reader.read(filename_queue)
        record_defaults = [[1], ['']]
        col1, col2 = tf.decode_csv(value_temp, record_defaults=record_defaults)

        file_content = tf.read_file(col2)
        train_image = tf.image.decode_png(file_content, channels=self.num_channels)
        if self.whetherTrain:
            print("whetherTrain is ture, do pad, flip and crop")
            train_image = tf.image.resize_image_with_pad(train_image, self.image_width + self.pad, self.image_width + self.pad)
            train_image = tf.image.random_flip_left_right(train_image)
        train_image = tf.random_crop(train_image, [self.image_height, self.image_width, 3], seed=self.seed, name="crop")
        train_image = tf.image.per_image_standardization(train_image)

        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * self.batch_size
        self.example_batch, self.label_batch = tf.train.shuffle_batch (
                [train_image, col1], batch_size=self.batch_size, capacity=capacity,
                    min_after_dequeue=min_after_dequeue, seed=self.seed)

        return self.example_batch, self.label_batch
