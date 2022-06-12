import json
import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from keras import regularizers
from myGenerator import DataGenerator


class NeuralNetworkMLPNN:
    def __init__(self,train, truth_train, tc, val, truth_val, vc, spectra, op=1):
        self.trains = self.readInfo(train)
        self.id_trains = self.trains.id.unique().tolist()
        self.d_train = self.generate_discurse(tc)
        self.truth_train = self.generate_truth(truth_train)
        self.val = self.readInfo(val)
        self.d_val = self.generate_discurse(vc)
        self.id_val = self.val.id.unique().tolist()
        self.truth_val = self.generate_truth(truth_val)
        self.op = op
        self.spectra = self.readInfo(spectra)
        self.size = 2400 if op == 1 else 1208
        self.model = self.createModel()

    def train(self, epochs=100):
        params = {'dim': (self.size,),
                  'batch_size': 32,
                  'op': self.op}
        # Generators
        training_generator = DataGenerator(self.id_trains, self.trains, self.truth_train, 
                                            self.spectra, self.d_train, **params)
        validation_generator = DataGenerator(self.id_val, self.val, self.truth_val, 
                                            self.spectra, self.d_val, **params)
        # Train model on dataset with generator.
        self.history = self.model.fit(training_generator,
                                 validation_data = validation_generator, 
                                 workers = 1,
                                 use_multiprocessing=True, epochs = epochs)

    def createModel(self):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.size),
            tf.keras.layers.Dense(1200, activation='relu',kernel_regularizer=regularizers.l2(0.001), input_shape=(1200,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(800, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(250, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        ])

        opt = tf.keras.optimizers.Adam(learning_rate=.0003)

        model.compile(optimizer=opt,
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=['accuracy'])

        # This callback will stop the training when there is no improvement in
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        
        print(model.summary())
        return model
    
    def save(self, path):
        return self.model.save(path)

    def readInfo(self, path):
        infile = open(path,'rb')
        file = pickle.load(infile)
        infile.close()
        return file
    
    def generate_truth(self, truth_set):
        aux = {}
        with open(truth_set) as infile:
            for line in infile:
                val = json.loads(line)
                aux[val["id"]] = int(val["value"])
        return aux
    
    def generate_discurse(self, data):
        aux = {}
        cat = {"email":1, "text_message":2, "essay": 3, "memo":4}
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit([["email",1], ["text_message",2], ["essay", 3], ["memo",4]])
        with open(data) as infile:
            for line in infile:
                val = json.loads(line)
                dis = val["discourse_type"]
                dis = enc.transform([[dis[0], cat[dis[0]]], [dis[1], cat[dis[1]]]])
                aux[val["id"]] = dis.toarray()
        return aux
    
    def showHistory(self, param):
        plt.plot(self.history.history[param])
        plt.plot(self.history.history['val_' + param])
        plt.title('model ' + param)
        plt.ylabel(param)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('model_' + param + '.png')
        plt.show()

if __name__ == "__main__":
    pXid = './pan22-authorship-verification-training-dataset/particionesXid/'
    p = './pan22-authorship-verification-training-dataset/particiones/'
    test = NeuralNetworkMLPNN(pXid+"PanTrain.plk", 
                              p+"train_truth.jsonl",
                              p+"train.jsonl",
                              pXid+"PanVal.plk", 
                              p+"val_truth.jsonl", 
                              p+"val.jsonl",
                              pXid+"FullSpectra.plk", op=2)
    test.train()
    test.showHistory('accuracy')
    test.showHistory('loss')
    test.save('./model2022/discurso.h5')
    


