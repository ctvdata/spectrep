import json
import linecache
import pickle
from tensorflow import keras
import tensorflow as tf
import numpy as np


class corpusGenerator(keras.utils.Sequence):
    
    def __init__(self, corpus_set, truth_set, dim, option_test, 
                batch_size=32,):
        # 'Initialization'
        self.corpus_set = corpus_set
        self.truth_set = self.generate_truth(truth_set)
        self.batch_size = batch_size
        self.dim = dim
        self.option_test = option_test
        self.indexes = np.arange(len(self.corpus_set))
        self.path = r'./pan22-authorship-verification-training-dataset/particionesXid/FullSpectra.jsonl'

    def generate_truth(self, truth_set):
        aux = {}
        with open(truth_set) as infile:
            for line in infile:
                val = json.loads(line)
                aux[val["id"]] = int(val["value"])
        return aux

    def __len__(self):
        'Denotes the number of batches per epoch'
        #Tomaremos por parejas los documentos
        return int(np.floor(len(self.corpus_set) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # id, idtext  of the batch (batch tiene que ser par para leer de dos en dos)
        batch = self.corpus_set[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data X (entrenamiento), y (valores de verdad)
        X, y = self.__data_generation(batch)
        return X, y

    # Prueba 1: Flatten y Concatenación de vectores.
    def test1(self, s1, s2):
        s1 = s1.flatten()
        s2 = s2.flatten()
        return np.concatenate([s1,s2])

    # Prueba 2: Resta de matrices y flatten.
    def test2(self,s1, s2):
        aux = s1 - s2
        return aux.flatten()

    # Prueba 3: Resta de matrices, flatten y softmax.
    def softmax(self, aux):
        return np.exp(aux)/np.sum(np.exp(aux))
    
    def get_spectre(self, id):
        spectre = linecache.getline(self.path, id+1)
        spectre = json.loads(spectre)
        return np.array(spectre["spectra"])

    def __data_generation(self, list_temp):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        ln = self.batch_size//2
        X = np.empty([ln, self.dim])
        y = np.empty(ln)

        # Generate data
        for i in range(0,ln):
            j = i*2
            # Espectro del documento
            s1 = self.get_spectre(list_temp.iloc[j]['idtext'])
            s2 = self.get_spectre(list_temp.iloc[j+1]['idtext'])

            if self.option_test == 1: 
                X[i,] = self.test1(s1, s2)
            else:
                X[i,] = self.test2(s1, s2)
                if self.option_test == 3:
                    X[i,] = self.softmax(X[i,])

            # Verdad del conjunto de problemas
            y[i] = self.truth_set[list_temp.iloc[j]['id']]

        return X, keras.utils.to_categorical(y)

class NeuralNetworkMLPNN:
    def __init__(self,train, truth_train, val, truth_val, op=1):
        self.trains = self.readInfo(train)
        self.truth_train = truth_train
        self.val = self.readInfo(val)
        self.truth_val = truth_val
        self.op = op
        self.size = 1200 if op == 2 else 2400
        self.model = self.createModel()

    def train(self, epochs=100):
        # Generators
        training_generator = corpusGenerator(self.trains, self.truth_train, self.size, self.op)
        validation_generator = corpusGenerator(self.val, self.truth_val, self.size, self.op)
        # Train model on dataset with generator.
        self.model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,epochs=epochs)

    def createModel(self):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.size),
            tf.keras.layers.Dense(600, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
        print(model.summary())
        return model
    
    def save(self, path):
        return self.model.predict(path)

    def readInfo(self, path):
        infile = open(path,'rb')
        file = pickle.load(infile)
        infile.close()
        return file

if __name__ == "__main__":
    pXid = './pan22-authorship-verification-training-dataset/particionesXid/'
    p = './pan22-authorship-verification-training-dataset/particiones/'
    test = NeuralNetworkMLPNN(pXid+"PanTrain.plk", 
                              p+"train_truth.jsonl",
                              pXid+"PanVal.plk", 
                              p+"val_truth.jsonl", op=2)
    test.train()
    test.save('./model2022/test2.h5')


