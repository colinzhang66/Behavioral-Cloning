import os
import random

from pandas import read_csv
import numpy as np

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D, ELU
from keras.layers.convolutional import Convolution2D
from keras.callbacks import EarlyStopping

from scipy.misc import imread, imsave
from sklearn.model_selection import train_test_split

def load_data(path):
    csv_data = read_csv(path)
    print('Loaded driving log data of length ', len(csv_data))
    print('Split driving log into Training data and Validation data')
    train_data, valid_data = train_test_split(csv_data, test_size = 0.15)
    print('Length of Training Data ', len(train_data))
    print('Length of Validation Data ', len(valid_data))
    return train_data, valid_data

def save_model(model):
    os.remove('model.h5') if os.path.exists('model.h5') else None
    model.save('model.h5')
    print('Model Saved.')

def random_augment_image_measurement(image, measurement):
    """
    Augments the image and measurement. 
    Flips the image in the right direction.
    Negates the measurement.
    """
    if random.random() > 0.5:
        return np.fliplr(image), -measurement
    return image, measurement

def get_image_and_measurement(idx, data):
    '''
    Fetches the images and steering angle from index (index)
    Randomly chooses between the center, right, left image.
    Randomly adds corrections to the steering angle
    '''
    image_orientations = ['left', 'center', 'right']
    steering_delta = [0.25, 0.0, -0.25]
    random_index = random.choice([0, 1, 2])
    # Get the index of the element in the data set
    index = data.index[idx]
    # Pick the steering angle at index and randomly add a delta to it
    measurement = data['steering'][index] + steering_delta[random_index]
    # Randomly pick any of the 3 images i.e. left right center from the index
    image = imread(data[image_orientations[random_index]][index])
    return random_augment_image_measurement(image, measurement)

def generate_samples(data, batch_size):
    '''
    Generates the samples for training and validation data for the model generator
    '''
    while True:
        for start in range(0, len(data), batch_size):
            images, measurements = [], []
            for i in range(start, start + batch_size):
                if i >= len(data):
                    continue
                image, measurement = get_image_and_measurement(i, data)
                images.append(image)
                measurements.append(measurement)
            yield np.array(images), np.array(measurements)

def create_model(params):
    '''
    Create the model
    '''
    model = Sequential()
    # Crop 70 pixels from the top of the image and 25 from the bottom
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=params['input_shape']))
    # Normalize the data
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=params['input_shape']))
    # Conv layer 1
    model.add(Convolution2D(params['filter_size'], params['initial_conv2d_row'], params['initial_conv2d_col'], subsample=params['initial_subsample'], border_mode=params['border_mode']))
    model.add(ELU())
    # Conv layer 2
    model.add(Convolution2D(params['filter_size']*2, params['final_conv2d_row'], params['final_conv2d_col'], subsample=params['final_subsample'], border_mode=params['border_mode']))
    model.add(ELU())
    # Conv layer 3
    model.add(Convolution2D(params['filter_size']*4, params['final_conv2d_row'], params['final_conv2d_col'], subsample=params['final_subsample'], border_mode=params['border_mode']))
    model.add(Flatten())
    model.add(Dropout(params['first_dropout']))
    model.add(ELU())
    # Fully connected layer 1
    model.add(Dense(params['first_dense_units']))
    model.add(Dropout(params['second_dropout']))
    model.add(ELU())
    # Fully connected layer 2
    model.add(Dense(params['second_dense_units']))
    # Summarize the model in the terminal
    model.summary()
    return model

def train(params):
    """
    Load data and train model
    """
    model = create_model(params)
    model.compile(optimizer=Adam(lr=params['learning_rate']), loss = 'mse', metrics=params['metrics'])
    train_data, valid_data = load_data('driving_log.csv')
    print('Training model for shape : ', params['input_shape'])
    # Early stopping function if the accuracy is not increasing anymore then stop the NN
    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience = 1, 
                                   min_delta = 0.00020,
                                   verbose = params['verbose'], 
                                   mode = 'auto')
    model.fit_generator(generator=generate_samples(train_data, batch_size=params['batch_size']),
                        samples_per_epoch = len(train_data),
                        validation_data = generate_samples(valid_data, batch_size=params['batch_size']),
                        nb_val_samples = len(valid_data),
                        nb_epoch = params['epochs'],
                        callbacks = [early_stopping],
                        verbose = params['verbose'])
    print('Succesfully Model Trained')
    save_model(model)

params = {
  'input_shape' : (160, 320, 3),
  'border_mode' : 'same',
  'filter_size' : 16,
  'initial_subsample' : (4,4),
  'final_subsample' : (2,2),
  'initial_conv2d_row' : 8,
  'initial_conv2d_col' : 8,
  'final_conv2d_row' : 5,
  'final_conv2d_col' : 5,
  'learning_rate' : 0.001,
  'first_dropout' : 0.2,
  'second_dropout' : 0.5,
  'first_dense_units' : 512,
  'second_dense_units' : 1,
  'metrics' : ['accuracy'],
  'epochs' : 10,
  'batch_size' : 64,
  'verbose' : 1
}

if __name__ == '__main__':
    train(params)