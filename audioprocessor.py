
import os

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils


class Processor():
    def __init__(self, data_dir, train_csv, test_csv):
        self.data_dir = data_dir
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_csv = train_csv
        self.pickle_train = './train_df.pkl'
        self.pickle_test = './test_df.pkl'
        self.train_df = None
        self.test_df = None
        self.test_csv = test_csv
        self.history = None
        self.acc = None
        self.val_acc = None
        self.loss = None
        self.val_loss = None
        self.model = None
        self.lb = None
        self.predictions = None

    def process_training(self, test_size=0.25, random_state=23):
        if os.path.isfile(self.pickle_train):
            print('Loading from pickle, {}'.format(self.pickle_train))
            self.train_df = pd.read_pickle(self.pickle_train)
        else:
            train = pd.read_csv(self.data_dir + self.train_csv)

            self.train_df = pd.DataFrame(train.apply(self.train_parser, axis=1))
            self.train_df.rename(columns={0:'Features'}, inplace=True)
            self.train_df['Label'] = self.train_df['Features'].map(lambda x: x[1])
            self.train_df['Features'] = self.train_df['Features'].map(lambda x: x[0])
            self.train_df.to_pickle(self.pickle_train)

        target = self.train_df['Label']
        features = self.train_df.drop('Label', axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, target, test_size=0.25, random_state=23)


    def train_parser(self, row):
       # function to load files and extract features
        file_name = os.path.join(os.path.abspath(self.data_dir), 'Train', str(row.ID) + '.wav')
       # handle exception to check if there isn't a file which is corrupted
        try:
    #       # here kaiser_fast is a technique used for faster extraction
            X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    #       # we extract mfcc feature from data
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
            return None

        feature = mfccs
        label = row.Class

        return [feature, label]

    def test_parser(self, row):
       # function to load files and extract features
        file_name = os.path.join(os.path.abspath(self.data_dir), 'Test', str(row.ID) + '.wav')
       # handle exception to check if there isn't a file which is corrupted
        try:
    #       # here kaiser_fast is a technique used for faster extraction
            X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    #       # we extract mfcc feature from data
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
            return None

        return [mfccs, row.ID]

    def prep_x_test(self):
        if os.path.isfile(self.pickle_test):
            print('Loading from pickle, {}'.format(self.pickle_test))
            self.test_df = pd.read_pickle(self.pickle_test)
        else:
            test = pd.read_csv(self.data_dir + self.test_csv)
            self.test_df = pd.DataFrame(test.apply(self.test_parser, axis=1))
            # self.test_df.rename(columns={0:'Features'}, inplace=True)
            self.test_df.rename(columns={0:'Features'}, inplace=True)
            self.test_df['ID'] = self.test_df['Features'].map(lambda x: x[1])
            self.test_df['Features'] = self.test_df['Features'].map(lambda x: x[0])
            self.test_df.to_pickle(self.pickle_test)



    def show_accuracy(self):
        plt.figure(figsize=(18, 10))
        acc = list(self.history.history['acc'])
        val_acc = list(self.history.history['val_acc'])
        plt.plot(acc)
        plt.plot(val_acc)
        plt.title('model_accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        plt.savefig('output/accuracy.png')
        plt.show()

    def show_loss(self):
        plt.figure(figsize=(18, 10))
        loss = list(self.history.history['loss'])
        val_loss = list(self.history.history['val_loss'])
        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('model_loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        plt.savefig('output/loss.png')
        plt.show()

    def predict (self):
        # X_predict = np.array(self.test_df.Features.tolist())
        # predictions = self.model.predict_classes(X_predict)
        # self.test_df['prediction_number'] = predictions
        # self.test_df['prediction_label'] = self.lb.inverse_transform(predictions)

        self.test_df['prediction_number'] = self.test_df.Features.map(lambda x: self.model.predict_classes(x))
        self.test_df['prediction_label'] = self.test_df.prediction_number.map(lambda x: self.lb.inverse_transform(x))
        # self.lb.inverse_transform(self.predictions))
        # print(self.predictions)
        self.test_df.to_pickle('./predictions_on_test.pkl')


    def run(self, num_epochs=50):

        X_train = np.array(self.X_train.Features.tolist())
        y_train = np.array(self.y_train.tolist())

        X_test = np.array(self.X_test.Features.tolist())
        y_test = np.array(self.y_test.tolist())

        lb = LabelEncoder()
        self.lb = lb

        y_train = np_utils.to_categorical(lb.fit_transform(y_train))
        y_test = np_utils.to_categorical(lb.fit_transform(y_test))

        num_labels = y_train.shape[1]
        filter_size = 2

        # build model
        model = Sequential()

        model.add(Dense(256, input_shape=(40,)))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(num_labels))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        self.history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), shuffle=False, verbose=0)
        self.model = model
        self.acc = self.history.history['acc']
        self.val_acc = self.history.history['val_acc']
        self.loss = self.history.history['loss']
        self.val_loss = self.history.history['val_loss']

        return self.acc, self.val_acc, self. loss, self.val_loss
