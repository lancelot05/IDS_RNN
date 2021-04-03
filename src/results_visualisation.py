import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_score, auc)

from kdd_processing import kdd_encoding

params = {'train_data': 494021, 'features_nb': 4,
          'batch_size': 1024, 'encoder': 'standarscaler',
          'dataset': 'kdd'}

# Change as per model name
model_name = './models/' + '494021_4_mse_nadam_sigmoid_1_128_1024' + \
    '_0.2_CuDNNLSTM_standarscaler_1562685990.8704927st'


def load_data():
    x_train, x_test, y_train, y_test = kdd_encoding(params)
    # Reshape the inputs in the accepted model format
    x_train = np.array(x_train).reshape([-1, x_train.shape[1], 1])
    x_test = np.array(x_test).reshape([-1, x_test.shape[1], 1])
    return x_train, x_test, y_train, y_test

# Print information on the results


def print_results(params, model, x_train, x_test, y_train, y_test):
    print("Val loss and acc:")
    print(model.evaluate(x_test, y_test, params['batch_size']))

    y_pred = model.predict(x_test, params['batch_size'])

    print('\nConfusion Matrix:')
    conf_matrix = confusion_matrix(
        y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(conf_matrix)

    print('\nPrecision:')  # Probability an instance gets correctly predicted

    print(precision_score(y_true=y_test.argmax(axis=1),
                          y_pred=y_pred.argmax(axis=1), average=None))


if __name__ == "__main__":
    # Allows tensorflow to run multiple sessions and learning simultaneously
    # Comment the 3 following lines if causing issues
    # config = tensorflow.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tensorflow.Session(config=config)

    model = load_model(model_name)
    model.summary()

    x_train, x_test, y_train, y_test = load_data()
    print_results(params, model, x_train, x_test, y_train, y_test)
