import tensorflow as tf
from keras.backend.common import epsilon
from tensorflow.python.ops import ctc_ops as ctc
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import CuDNNLSTM
from keras.layers import Bidirectional
from keras.layers import GRU
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Activation
from keras.layers import concatenate
from keras.layers import add
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K


def ctc_batch_cost(y_true, y_pred, input_length, label_length,
                            ctc_merge_repeated=False):
    label_length = tf.cast(tf.squeeze(label_length, axis=-1), tf.int32)
    input_length = tf.cast(tf.squeeze(input_length, axis=-1), tf.int32)
    sparse_labels = tf.cast(
        K.ctc_label_dense_to_sparse(y_true, label_length), tf.int32)
    y_pred = tf.log(tf.transpose(y_pred, perm=[1, 0, 2]) + epsilon())
    return tf.expand_dims(ctc.ctc_loss(inputs=y_pred,
                                       labels=sparse_labels,
                                       sequence_length=input_length,
                                       ctc_merge_repeated=ctc_merge_repeated), 1)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return ctc_batch_cost(labels, y_pred, input_length, label_length)


def best_model():
    input_data = Input(shape=(None, 26), name='input')

    x = Conv1D(128, 3, strides=1, padding='same', activation='relu',
                     name='conv_1')(input_data)
    x = Dropout(0.1, name='dropout_1')(x)
    x = Conv1D(128, 3, strides=1, padding='same', activation='relu',
                     name='conv_2')(x)
    x = Dropout(0.1, name='dropout_2')(x)
    x = Conv1D(128, 3, strides=1, padding='same', activation='relu',
                     name='conv_3')(x)
    x = Dropout(0.1, name='dropout_3')(x)

    x = TimeDistributed(Dense(512, activation='relu'))(x)
    x = Dropout(0.1, name='dropout_4')(x)

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True), merge_mode='sum')(x)

    x = TimeDistributed(Dense(512, activation='relu'))(x)
    x = Dropout(0.1, name='dropout_5')(x)
    y_pred = TimeDistributed(Dense(39 + 1, activation='softmax'))(x)

    # Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
                        [y_pred, labels, input_length, label_length])

    adam = Adam()

    model = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=loss_out)

    # captures output of softmax so we can decode the output during
    # visualization
    test_func = K.function([input_data], [y_pred])

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

    #return model, model_p, test_func
    return model, test_func
