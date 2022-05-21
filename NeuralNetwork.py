import json
import pickle
from matplotlib import pyplot as plt
import tensorflow as tf
from myGenerator import DataGenerator


class NeuralNetworkMLPNN:
    def __init__(self,train, truth_train, val, truth_val, spectra, op=1):
        self.trains = self.readInfo(train)
        self.id_trains = self.trains.id.unique().tolist()
        self.truth_train = self.generate_truth(truth_train)
        self.val = self.readInfo(val)
        self.id_val = self.val.id.unique().tolist()
        self.truth_val = self.generate_truth(truth_val)
        self.op = op
        self.spectra = self.readInfo(spectra)
        self.size = 2400 if op == 1 else 1200
        self.model = self.createModel()

    def train(self, epochs=100):
        params = {'dim': (self.size,),
                  'batch_size': 32,
                  'op': self.op}
        # Generators
        training_generator = DataGenerator(self.id_trains, self.trains, self.truth_train, 
                                            self.spectra, **params)
        validation_generator = DataGenerator(self.id_val, self.val, self.truth_val, 
                                            self.spectra, **params)
        # Train model on dataset with generator.
        self.history = self.model.fit(training_generator,
                                 validation_data = validation_generator, 
                                 workers = 1,
                                 use_multiprocessing=True, epochs = epochs)

    def createModel(self):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.size),
            tf.keras.layers.Dense(600, activation='relu'),
            tf.keras.layers.Dense(2, activation='sigmoid'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
                        metrics=[tf.keras.metrics.BinaryAccuracy()])
        print(model.summary())
        return model
    
    def save(self, path):
        # save the model to disk
        pickle.dump(self.model, open(path+".pkl", 'wb'))
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
                aux[val["id"]] = tf.one_hot(int(val["value"]), 2)
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
                              pXid+"PanVal.plk", 
                              p+"val_truth.jsonl", 
                              pXid+"FullSpectra.plk", op=3)
    test.train()
    test.showHistory('binary_accuracy')
    test.showHistory('loss')
    test.save('./model2022/test3.h5')


