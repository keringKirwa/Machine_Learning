import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
import logging
"""Note the role of the tensorflow in this case is  to allow us get an image from the FileSystem , which=ch is 
actuality stored in form of a matrix: the image flows in from the memory in form of a matrix , to manage this then we 
use tensorflow."""


EPOCHS = 10
input_shape = (150, 150, 3)

"""
Note that for the rescale , each  image pixel  value is multiplied by (1/255) to rescale it.
"""
train_dir = '/home/arapkering/Desktop/dogs-vs-cats/train'

if __name__ == '__main__':

    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    if not os.path.exists((os.path.join(train_dir, 'cat'))):
        os.mkdir(os.path.join(train_dir, 'cat'))
    if not os.path.exists((os.path.join(train_dir, 'dog'))):
        os.mkdir(os.path.join(train_dir, 'dog'))

    BATCH_SIZE = 100  # Number of training examples to process before updating our models variables
    IMG_SHAPE = 150

    image_generator = ImageDataGenerator(rescale=1. / 255, validation_split=0.4)

    """
    (1)The train_data_train_data_gen function returns an iterator on images. 
    (2)Note that we had already defined how the data splitting will be done , and henge by using just one generator ,  
    in every batch , we get 80 % as the  training data and he other 20 % as the testing data.
    (4)All of these images have labels :Note they are read from the same dir but different sub_directories
    (5)A single batch contains both dogs and cats.
    """

    train_data_gen = image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                         directory=train_dir,
                                                         shuffle=True,
                                                         target_size=(IMG_SHAPE, IMG_SHAPE),  # (150,150)
                                                         class_mode='binary',
                                                         subset='training')

    val_data_gen = image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                       directory=train_dir,
                                                       shuffle=False,
                                                       target_size=(IMG_SHAPE, IMG_SHAPE),  # (150,150)
                                                       class_mode='binary',
                                                       subset='validation')

    # the _ is used to ignore the labels Very IMPORTANT .

    sample_training_images, _ = next(train_data_gen)  # gets the FIRST batch(first 200 images only) in the generator ,


    def plotImages(images_arr):
        # the n_rows and the n_columns determine the number of  subplots to be made.(1*5 in the case plots 5 objects)
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 20), sharex="none", sharey="none")

        for singleImageArray, axisObject in zip(images_arr, axes):
            axisObject.imshow(singleImageArray)
        plt.tight_layout()
        plt.show()


    plotImages(sample_training_images[:5])

    """Note that for image processing , we dont want any negative value , to do this , we use the rectified LineaR 
    Unit(ReLU ) activation  function on every convolution Layer to keep the  positive values while setting the 
    negatives as zeros """

    # Define the layers
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
    pool1 = layers.MaxPooling2D(2, 2)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu')
    pool2 = layers.MaxPooling2D(2, 2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu')
    pool3 = layers.MaxPooling2D(2, 2)
    conv4 = layers.Conv2D(128, (3, 3), activation='relu')
    pool4 = layers.MaxPooling2D(2, 2)
    flatten = layers.Flatten()
    dense1 = layers.Dense(512, activation='relu')
    dense2 = layers.Dense(2)

    # Define the model
    model = models.Sequential([
        conv1,
        pool1,
        conv2,
        pool2,
        conv3,
        pool3,
        conv4,
        pool4,
        flatten,
        dense1,
        dense2
    ])
    """
    model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adamax(lr=0.001),
              metrics=['acc'])"""

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    """(1)Note that the fit_generator is the one responsible for  getting  the next batch og image tensors , 
    it calls the next() function . (2)The  fit() or the fit_generator will be fetching  data from the 
    train_data_generator() steps_per_epoch == 200 , that means that for one epoch to be complete , then 200 batches @ 
    with 100 images must have been attended to ."""

    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=int(np.ceil(20000 / float(BATCH_SIZE))),
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps=int(np.ceil(5000 / float(BATCH_SIZE))),
        verbose=2
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # img = mpimg.imread(dataset_dir + '11.jpg')
    # # Show image using Matplotlib
    # plt.imshow(img)
    # plt.show()
