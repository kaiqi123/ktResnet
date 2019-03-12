import numpy as np
from keras.datasets import cifar10
from scipy.misc import imresize

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, Flatten
from keras import backend as K

seed = 42
epochs = 10
records_per_class = 100

# We take only 2 classes from CIFAR10 and a very small sample to intentionally overfit the model.
# We will also use the same data for train/test and expect that Keras will give the same accuracy.
(x, y), _ = cifar10.load_data()


def filter_resize(category):
    # We do the preprocessing here instead in the Generator to get around a bug on Keras 2.1.5.
    return [preprocess_input(imresize(img, (224, 224)).astype('float')) for img in
            x[y.flatten() == category][:records_per_class]]


x = np.stack(filter_resize(3) + filter_resize(5))
records_per_class = x.shape[0] // 2
y = np.array([[1, 0]] * records_per_class + [[0, 1]] * records_per_class)

# We will use a pre-trained model and finetune the top layers.
np.random.seed(seed)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
l = Flatten()(base_model.output)
predictions = Dense(2, activation='softmax')(l)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:140]:
    layer.trainable = False

for layer in model.layers[140:]:
    layer.trainable = True

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(ImageDataGenerator().flow(x, y, seed=42), epochs=epochs,
                    validation_data=ImageDataGenerator().flow(x, y, seed=42))

# Store the model on disk
model.save('tmp.h5')

# In every test we will clear the session and reload the model to force Learning_Phase values to change.
print('DYNAMIC LEARNING_PHASE')
K.clear_session()
model = load_model('tmp.h5')
# This accuracy should match exactly the one of the validation set on the last iteration.
print(model.evaluate_generator(ImageDataGenerator().flow(x, y, seed=42)))

print('STATIC LEARNING_PHASE = 0')
K.clear_session()
K.set_learning_phase(0)
model = load_model('tmp.h5')
# Again the accuracy should match the above.
print(model.evaluate_generator(ImageDataGenerator().flow(x, y, seed=42)))

print('STATIC LEARNING_PHASE = 1')
K.clear_session()
K.set_learning_phase(1)
model = load_model('tmp.h5')
# The accuracy will be close to the one of the training set on the last iteration.
print(model.evaluate_generator(ImageDataGenerator().flow(x, y, seed=42)))