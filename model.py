import csv
import os

import numpy as np
import sklearn
from keras.layers import (Convolution2D, Cropping2D, Dense, Dropout, Flatten,
                          Lambda)
from keras.models import Sequential
from sklearn.model_selection import train_test_split

import cv2

DATA_DIR = os.path.join(os.getcwd(), 'data')
HYPER_PARAM = {
    'batch_size': 128,
    'create_mirror_image': True,
    'epochs': 3,
    'loss': 'mse',
    'optimizer': 'adam',
    'steering_correction': 0.2,
    'test_size': 0.2,
}


def get_samples():
    samples = []
    with open(os.path.join(DATA_DIR, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    return samples


def create_steering_correction(steering_angle):
    """Create adjusted steering measurements for the side camera images"""
    steering_left = steering_angle + HYPER_PARAM['steering_correction']
    steering_right = steering_angle - HYPER_PARAM['steering_correction']
    return steering_left, steering_right


def extract_images_and_measurement(line):
    """Extract the images and measures ments from a line in the csv file"""
    def extract_filename(a): return os.path.join(
        DATA_DIR, 'IMG', os.path.split(a)[-1])

    center_image = cv2.imread(extract_filename(line[0]))
    left_image = cv2.imread(extract_filename(line[1]))
    right_image = cv2.imread(extract_filename(line[2]))
    measurement = float(line[3])
    return center_image, left_image, right_image, measurement


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Extract all images and steering angle
                center, left, right, steering_angle = extract_images_and_measurement(
                    batch_sample)
                # Create corrected steering angles for the left and right images
                steering_left, steering_right = create_steering_correction(
                    steering_angle)

                for img in [center, left, right]:
                    images.append(img)
                    if HYPER_PARAM['create_mirror_image']:
                        images.append(np.fliplr(img))

                for angle in [steering_angle, steering_left, steering_right]:
                    angles.append(angle)
                    if HYPER_PARAM['create_mirror_image']:
                        angles.append(-angle)

            yield sklearn.utils.shuffle(np.array(images), np.array(angles))


def nvidia_model():
    """Keras implementation of the Nvidia end-to-end model"""
    (row, col, ch) = (160, 320, 3)  # Image size
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


if __name__ == '__main__':
    # Get the image paths, and steering angles
    samples = get_samples()

    extract_images_and_measurement(samples[0])

    # Split the dataset into training and validation sets
    train_samples, validation_samples = train_test_split(
        samples, test_size=HYPER_PARAM['test_size'])

    # Create generators to dynamically load the data during training,
    # this reduces the memory usage needed to train the model
    train_generator = generator(train_samples,
                                HYPER_PARAM['batch_size'])
    validation_generator = generator(validation_samples,
                                     HYPER_PARAM['batch_size'])

    # Calculate the sizes of the validation and training sets
    if HYPER_PARAM['create_mirror_image']:
        nb_train_samples = len(train_samples) * 6
        nb_val_samples = len(validation_samples) * 6
    else:
        nb_train_samples = len(train_samples) * 3
        nb_val_samples = len(validation_samples) * 3

    # Create the Nvidia model, define the training operation
    # and start the training
    model = nvidia_model()
    model.compile(loss=HYPER_PARAM['loss'], optimizer=HYPER_PARAM['optimizer'])
    model.fit_generator(train_generator,
                        samples_per_epoch=nb_train_samples,
                        validation_data=validation_generator,
                        nb_val_samples=nb_val_samples,
                        nb_epoch=HYPER_PARAM['epochs'])

    # Save the trained model
    model.save('model.h5')
