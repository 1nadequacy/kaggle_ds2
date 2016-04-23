from __future__ import print_function

from keras.models import Sequential, weighted_objective
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Flatten, Dropout, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta, SGD
from keras.regularizers import l2
from keras import backend as K

import theano.tensor as T
from keras import optimizers
from keras import objectives


def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return x #(x - K.mean(x)) / K.std(x)


def get_model(num_frames=30, shape_x=74, shape_y=74, out=1, d=32):
    model = Sequential()
    model.add(Activation(activation=center_normalize,
                         input_shape=(num_frames, shape_x, shape_y)))

    model.add(Convolution2D(d, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(d, 3, 3, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(2 * d, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(2 * d, 3, 3, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(4 * d, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(4 * d, 2, 2, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, W_regularizer=l2(1e-3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out))
    if out != 1:
        model.add(Activation('sigmoid'))

    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='mse')
    return model


def get_leaky_model(num_frames=30, shape_x=74, shape_y=74, out=1, d=32):
    model = Sequential()
    model.add(Activation(activation=center_normalize,
                         input_shape=(num_frames, shape_x, shape_y)))

    model.add(Convolution2D(d, 3, 3, border_mode='same'))
    model.add(LeakyReLU())
    model.add(Convolution2D(d, 3, 3, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(2 * d, 3, 3, border_mode='same'))
    model.add(LeakyReLU())
    model.add(Convolution2D(2 * d, 3, 3, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(4 * d, 2, 2, border_mode='same'))
    model.add(LeakyReLU())
    model.add(Convolution2D(4 * d, 2, 2, border_mode='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, W_regularizer=l2(1e-3)))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(out))
    if out != 1:
        model.add(Activation('sigmoid'))

    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='mse')
    return model


def get_lenet(num_frames=30, shape_x=74, shape_y=74, out=1):
    model = Sequential()
    model.add(Activation(lambda x: x,
                         input_shape=(num_frames, shape_x, shape_y)))

    model.add(Convolution2D(40, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(40, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, W_regularizer=l2(1e-3)))
    model.add(Activation('relu'))
    model.add(Dense(out))
    if out != 1:
        model.add(Activation('sigmoid'))

    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='mse')
    return model


def get_full(num_frames=30, shape_x=74, shape_y=74, out=1, d=16):
    model = MySequential()
    model.add(Activation(activation=center_normalize,
                         input_shape=(num_frames, shape_x, shape_y)))

    model.add(Convolution2D(d, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(d, 3, 3, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(2 * d, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(2 * d, 3, 3, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(4 * d, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(4 * d, 2, 2, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, W_regularizer=l2(1e-3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out))
    if out != 1:
        model.add(Activation('sigmoid'))

    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='mse')
    return model


class MySequential(Sequential):
    def compile(self, optimizer, loss,
                class_mode="categorical",
                sample_weight_mode=None):
        '''Configure the learning process.

        # Arguments
            optimizer: str (name of optimizer) or optimizer object.
                See [optimizers](optimizers.md).
            loss: str (name of objective function) or objective function.
                See [objectives](objectives.md).
            class_mode: one of "categorical", "binary".
                This is only used for computing classification accuracy or
                using the predict_classes method.
            sample_weight_mode: if you need to do timestep-wise
                sample weighting (2D weights), set this to "temporal".
                "None" defaults to sample-wise weights (1D).
        '''
        self.optimizer = optimizers.get(optimizer)
        self.sample_weight_mode = sample_weight_mode

        self.loss = objectives.get(loss)
        weighted_loss = weighted_objective(self.loss)

        # input of model
        self.X_train = self.get_input(train=True)
        self.X_test = self.get_input(train=False)

        self.single_y_train = self.get_output(train=True)
        self.single_y_test = self.get_output(train=False)

        self.diff_train = K.placeholder(ndim=1)
        self.diff_test = K.placeholder(ndim=1)

        self.y_train = K.dot(self.diff_train, self.single_y_train)
        self.y_test = K.dot(self.diff_test, self.single_y_test)

        # target of model
        self.y = K.placeholder(ndim=K.ndim(self.y_train))

        if self.sample_weight_mode == 'temporal':
            self.weights = K.placeholder(ndim=2)
        else:
            self.weights = K.placeholder(ndim=1)

        if hasattr(self.layers[-1], "get_output_mask"):
            mask = self.layers[-1].get_output_mask()
        else:
            mask = None
        train_loss = weighted_loss(self.y, self.y_train, self.weights, mask)
        test_loss = weighted_loss(self.y, self.y_test, self.weights, mask)

        if class_mode == "categorical":
            train_accuracy = K.mean(K.equal(K.argmax(self.y, axis=-1),
                                            K.argmax(self.y_train, axis=-1)))
            test_accuracy = K.mean(K.equal(K.argmax(self.y, axis=-1),
                                           K.argmax(self.y_test, axis=-1)))

        elif class_mode == "binary":
            train_accuracy = K.mean(K.equal(self.y, K.round(self.y_train)))
            test_accuracy = K.mean(K.equal(self.y, K.round(self.y_test)))
        else:
            raise Exception("Invalid class mode:" + str(class_mode))
        self.class_mode = class_mode

        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.trainable_weights,
                                             self.constraints,
                                             train_loss)
        updates += self.updates

        if type(self.X_train) == list:
            train_ins = self.X_train + [self.diff_train, self.y, self.weights]
            test_ins = self.X_test + [self.diff_test, self.y, self.weights]
            assert type(self.X_test) == list
            predict_ins = self.X_test + [self.diff_test]
        else:
            train_ins = [self.X_train, self.diff_train, self.y, self.weights]
            test_ins = [self.X_test, self.diff_test, self.y, self.weights]
            predict_ins = [self.X_test, self.diff_test]

        self.__train = K.function(train_ins, [train_loss], updates=updates)
        self.__train_with_acc = K.function(train_ins, [train_loss, train_accuracy], updates=updates)
        self.__predict = K.function(predict_ins, [self.y_test], updates=self.state_updates)
        self.__test = K.function(test_ins, [test_loss], updates=self.state_updates)
        self.__test_with_acc = K.function(test_ins, [test_loss, test_accuracy], updates=self.state_updates)

        self._train = lambda rr: self.__train([r[0] for r in rr[:-1]] + [rr[-1]])
        self._train_with_acc = lambda rr: self.__train_with_acc([r[0] for r in rr[:-1]] + [rr[-1]])
        self._predict = lambda rr: self.__predict([r[0] for r in rr])
        self._test = lambda rr: self.__test([r[0] for r in rr[:-1]] + [rr[-1]])
        self._test_with_acc = lambda rr: self.__test_with_acc([r[0] for r in rr[:-1]] + [rr[-1]])


class MySequential2(Sequential):
    def compile(self, optimizer, loss,
                class_mode="categorical",
                sample_weight_mode=None):
        '''Configure the learning process.

        # Arguments
            optimizer: str (name of optimizer) or optimizer object.
                See [optimizers](optimizers.md).
            loss: str (name of objective function) or objective function.
                See [objectives](objectives.md).
            class_mode: one of "categorical", "binary".
                This is only used for computing classification accuracy or
                using the predict_classes method.
            sample_weight_mode: if you need to do timestep-wise
                sample weighting (2D weights), set this to "temporal".
                "None" defaults to sample-wise weights (1D).
        '''
        self.optimizer = optimizers.get(optimizer)
        self.sample_weight_mode = sample_weight_mode

        self.loss = objectives.get(loss)
        weighted_loss = weighted_objective(self.loss)

        # input of model
        self.X_train = self.get_input(train=True)
        self.X_test = self.get_input(train=False)

        self.single_y_train = self.get_output(train=True)
        self.single_y_test = self.get_output(train=False)

        self.diff_train = K.placeholder(ndim=1)
        self.diff_test = K.placeholder(ndim=1)

        self.y_train = K.concatenate(
            [K.dot(self.diff_train,
                   self.single_y_train[:self.diff_train.shape[0]]),
             K.dot(self.diff_train,
                   self.single_y_train[self.diff_train.shape[0]:])],
            axis=0)
        self.y_test = K.concatenate(
            [K.dot(self.diff_test,
                   self.single_y_test[:self.diff_test.shape[0]]),
             K.dot(self.diff_test,
                   self.single_y_test[self.diff_test.shape[0]:])],
            axis=0)

        # target of model
        self.y = K.placeholder(ndim=K.ndim(self.y_train))

        if self.sample_weight_mode == 'temporal':
            self.weights = K.placeholder(ndim=2)
        else:
            self.weights = K.placeholder(ndim=1)

        if hasattr(self.layers[-1], "get_output_mask"):
            mask = self.layers[-1].get_output_mask()
        else:
            mask = None
        train_loss = weighted_loss(self.y, self.y_train, self.weights, mask)
        test_loss = weighted_loss(self.y, self.y_test, self.weights, mask)

        if class_mode == "categorical":
            train_accuracy = K.mean(K.equal(K.argmax(self.y, axis=-1),
                                            K.argmax(self.y_train, axis=-1)))
            test_accuracy = K.mean(K.equal(K.argmax(self.y, axis=-1),
                                           K.argmax(self.y_test, axis=-1)))

        elif class_mode == "binary":
            train_accuracy = K.mean(K.equal(self.y, K.round(self.y_train)))
            test_accuracy = K.mean(K.equal(self.y, K.round(self.y_test)))
        else:
            raise Exception("Invalid class mode:" + str(class_mode))
        self.class_mode = class_mode

        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.trainable_weights,
                                             self.constraints,
                                             train_loss)
        updates += self.updates

        if type(self.X_train) == list:
            train_ins = self.X_train + [self.diff_train, self.y, self.weights]
            test_ins = self.X_test + [self.diff_test, self.y, self.weights]
            assert type(self.X_test) == list
            predict_ins = self.X_test + [self.diff_test]
        else:
            train_ins = [self.X_train, self.diff_train, self.y, self.weights]
            test_ins = [self.X_test, self.diff_test, self.y, self.weights]
            predict_ins = [self.X_test, self.diff_test]

        self.__train = K.function(train_ins, [train_loss], updates=updates)
        self.__train_with_acc = K.function(train_ins, [train_loss, train_accuracy], updates=updates)
        self.__predict = K.function(predict_ins, [self.y_test], updates=self.state_updates)
        self.__test = K.function(test_ins, [test_loss], updates=self.state_updates)
        self.__test_with_acc = K.function(test_ins, [test_loss, test_accuracy], updates=self.state_updates)

        self._train = lambda rr: self.__train([r[0] for r in rr[:-1]] + [rr[-1]])
        self._train_with_acc = lambda rr: self.__train_with_acc([r[0] for r in rr[:-1]] + [rr[-1]])
        self._predict = lambda rr: self.__predict([r[0] for r in rr])
        self._test = lambda rr: self.__test([r[0] for r in rr[:-1]] + [rr[-1]])
        self._test_with_acc = lambda rr: self.__test_with_acc([r[0] for r in rr[:-1]] + [rr[-1]])
