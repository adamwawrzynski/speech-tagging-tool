from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import Bidirectional
from keras.layers import MaxPooling2D
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Masking
from keras.layers import Activation
from keras.models import Model
from keras import backend as K
from CTCModel import CTCModel

def fully_connected_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(26, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(61, activation='softmax'))
    return model

def simple_lstm_model():
    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(50, input_shape=(26, 1), return_sequences=False), merge_mode='concat'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(61, activation='softmax'))
    return model


def cnn_model():
    model = Sequential()
    model.add(Conv1D(32, (10), padding="same", activation="relu", input_shape=(26, 1)))
    model.add(MaxPooling1D(pool_size = (10), strides=(3)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(61, activation='softmax'))
    return model


def convnet():
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(5,3), activation='relu', input_shape=(None, 26)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))
    # model.add(Conv2D(256, kernel_size=(5,3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))
    model.add(Conv2D(384, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(384, kernel_size=(3,3), activation='relu'))
    # model.add(MaxPooling1D(pool_size=(2)))
    # model.add(Dropout(0.2))
    # model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(61, activation='softmax'))
    return model


def little_lstm():
    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True), 
                        input_shape=(None, 26), 
                        batch_input_shape=(1, None, 26), 
                        merge_mode='concat'))
    model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True), merge_mode='concat'))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(Dense(61, activation='softmax')))
    return model


def unidirectional_lstm():
    model = Sequential()
    model.add(CuDNNLSTM(1024, return_sequences=True,
                        input_shape=(None, 26)))
    model.add(CuDNNLSTM(1024, return_sequences=True))
    model.add(CuDNNLSTM(1024, return_sequences=True))
    model.add(TimeDistributed(Dense(61, activation='softmax')))
    return model

# def ctc_unidirectional_lstm():
#     input_layer = Input((None, 26))
#     lstm1 = CuDNNLSTM(10, return_sequences=True)(input_layer)
#     lstm2 = CuDNNLSTM(10, return_sequences=True)(lstm1)
#     lstm3 = CuDNNLSTM(10, return_sequences=True)(lstm2)
#     output_layer = TimeDistributed(Dense(61, activation='softmax'))(lstm3)
#     model = CTCModel([input_layer], [output_layer])
#     return model


def bigger_lstm():
    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(96, return_sequences=True), 
                        input_shape=(None, 26), 
                        batch_input_shape=(1, None, 26), 
                        merge_mode='concat'))
    model.add(Bidirectional(CuDNNLSTM(96, return_sequences=True), merge_mode='concat'))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(Dense(61, activation='softmax')))
    return model


def bigger_lstm2():
    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(96, return_sequences=True), 
                        input_shape=(None, 26), 
                        batch_input_shape=(1, None, 26), 
                        merge_mode='concat'))
    model.add(Bidirectional(CuDNNLSTM(96, return_sequences=True), merge_mode='concat'))
    model.add(Bidirectional(CuDNNLSTM(96, return_sequences=True), merge_mode='concat'))
    model.add(Bidirectional(CuDNNLSTM(96, return_sequences=True), merge_mode='concat'))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(Dense(61, activation='softmax')))
    return model


def bigger_lstm3():
    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(32, return_sequences=True), 
                        input_shape=(None, 26), 
                        batch_input_shape=(1, None, 26), 
                        merge_mode='concat'))
    model.add(Bidirectional(CuDNNLSTM(32, return_sequences=True), merge_mode='concat'))
    model.add(TimeDistributed(Dense(256, activation='relu')))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(Dense(61, activation='softmax')))
    return model


def new_cnn_lstm():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=9, strides=1, activation='relu', padding='same', input_shape=(None, 26)))
    model.add(TimeDistributed(Dropout(0.05)))
    model.add(Conv1D(filters=32, kernel_size=5, strides=1, activation='relu', padding='same'))
    model.add(TimeDistributed(Dropout(0.05)))
    model.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'))

    model.add(Bidirectional(CuDNNLSTM(96, return_sequences=True), merge_mode='concat'))
    model.add(Bidirectional(CuDNNLSTM(96, return_sequences=True), merge_mode='concat'))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(TimeDistributed(Dropout(0.05)))
    model.add(TimeDistributed(Dense(61, activation='softmax')))
    return model


def new_cnn_lstm2():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=9, strides=1, activation='relu', padding='same', input_shape=(None, 26)))
    model.add(TimeDistributed(Dropout(0.05)))
    model.add(Conv1D(filters=64, kernel_size=5, strides=1, activation='relu', padding='same'))
    model.add(TimeDistributed(Dropout(0.05)))
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))

    model.add(Bidirectional(CuDNNLSTM(96, return_sequences=True), merge_mode='concat'))
    model.add(Bidirectional(CuDNNLSTM(96, return_sequences=True), merge_mode='concat'))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(TimeDistributed(Dropout(0.05)))
    model.add(TimeDistributed(Dense(61, activation='softmax')))
    return model


def cnn_lstm():
    model = Sequential()
    model.add(Conv1D(32, kernel_size=5, strides=1, input_shape=(None, 26), activation='relu'))
    # x = TimeDistributed(Dropout(dropout), name='dropout_1')(x)
    model.add(Conv1D(32, kernel_size=2, strides=1, activation='relu'))
    model.add(Dropout(0.2))
    # x = TimeDistributed(Dropout(dropout), name='dropout_2')(x)
    model.add(Conv1D(32, kernel_size=2, strides=1, activation='relu'))
    model.add(Dropout(0.2))
    # x = TimeDistributed(Dropout(dropout), name='dropout_3')(x)
    model.add(Bidirectional(CuDNNLSTM(128, return_sequences=False), merge_mode='concat'))
    #model.add(Bidirectional(CuDNNLSTM(128, return_sequences=False), merge_mode='concat'))
    # 1 fully connected layer DNN ReLu with default 20% dropout
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    # x = TimeDistributed(Dropout(dropout), name='dropout_4')(x)

    # Output layer with softmax
    #model.add(TimeDistributed(Dense(61, activation='softmax')))
    model.add(Dense(61, activation='softmax'))
    return model



def cnn_blstm(units, input_dim=26, output_dim=61, dropout=0.2, n_layers=1):

    # Input data type
    dtype = 'float32'

    activation_conv = 'relu'

    # Kernel and bias initializers for fully connected dense layer
    kernel_init_dense = 'random_normal'
    bias_init_dense = 'random_normal'

    # Kernel and bias initializers for convolution layers
    kernel_init_conv = 'glorot_uniform'
    bias_init_conv = 'random_normal'

    # Kernel and bias initializers for recurrent layer
    kernel_init_rnn = 'glorot_uniform'
    bias_init_rnn = 'random_normal'

    # ---- Network model ----
    input_data = Input(name='the_input', shape=(None, 26), dtype=dtype)

    # 3 x 1D convolutional layers with strides: 1, 1, 2
    x = Conv1D(filters=units, kernel_size=5, strides=1, activation=activation_conv,
               kernel_initializer=kernel_init_conv, bias_initializer=bias_init_conv, name='conv_1')(input_data)
    # x = TimeDistributed(Dropout(dropout), name='dropout_1')(x)

    # x = Conv1D(filters=units, kernel_size=5, strides=1, activation=activation_conv,
    #            kernel_initializer=kernel_init_conv, bias_initializer=bias_init_conv, name='conv_2')(x)
    # x = TimeDistributed(Dropout(dropout), name='dropout_2')(x)

    # x = Conv1D(filters=units, kernel_size=5, strides=2, activation=activation_conv,
    #            kernel_initializer=kernel_init_conv, bias_initializer=bias_init_conv, name='conv_3')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_3')(x)

    # Bidirectional LSTM
    for i in range(0, n_layers):
        x = Bidirectional(CuDNNLSTM(units, kernel_initializer=kernel_init_rnn, bias_initializer=bias_init_rnn,
                                    unit_forget_bias=True, return_sequences=True),
                            merge_mode='sum', name='CuDNN_bi_lstm'+str(i+1))(x)

    # 1 fully connected layer DNN ReLu with default 20% dropout
    x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                              activation='relu'), name='fc_4')(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_4')(x)

    # Output layer with softmax
    y_pred = TimeDistributed(Dense(units=output_dim, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense, activation='softmax'), name='softmax')(x)

    # ---- CTC ----
    # y_input layers (transcription data) for CTC loss
    labels = Input(name='the_labels', shape=[None], dtype=dtype)       # transcription data (batch_size * y_seq_size)
    input_length = Input(name='input_length', shape=[1], dtype=dtype)  # unpadded len of all x_sequences in batch
    label_length = Input(name='label_length', shape=[1], dtype=dtype)  # unpadded len of all y_sequences in batch

    # Lambda layer with ctc_loss function due to Keras not supporting CTC layers
    loss_out = Lambda(function=ctc_lambda_func, name='ctc', output_shape=(61,))([y_pred, labels, input_length, label_length])

    network_model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    return network_model

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    #y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# Returns clipped relu, clip value set to 20.
def clipped_relu(value):
    return K.relu(value, max_value=20)
