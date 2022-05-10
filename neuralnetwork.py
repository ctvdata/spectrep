import json
import pickle
from tensorflow import keras
import tensorflow as tf
import numpy as np


class corpusGenerator(keras.utils.Sequence):
    
    def __init__(self, corpus_set, truth_set,dim, option_test, 
                batch_size=32,):
        # 'Initialization'
        self.corpus_set = corpus_set
        self.truth_set = self.generate_truth(truth_set)
        self.batch_size = batch_size
        self.dim = dim
        self.option_test = option_test
        self.indexes = np.arange(len(self.corpus_set))
        PATH = './pan22-authorship-verification-training-dataset/particionesXid/FullSpectra.jsonl'
        self.spectre = open(PATH,'rb')

    def generate_truth(self, truth_set):
        aux = {}
        with open(truth_set) as infile:
            for line in infile:
                val = json.loads(line)
                aux[val["id"]] = val["value"]
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
    def test_1(self, s1, s2):
        s1 = s1.flatten()
        s2 = s2.flatten()
        return np.concatenate([s1,s2])

    # Prueba 2: Resta de matrices y flatten.
    def test_2(self,s1, s2):
        aux = s1 - s2
        return aux.flatten()

    # Prueba 3: Resta de matrices, flatten y softmax.
    def softmax(self, aux):
        return np.exp(aux)/np.sum(np.exp(aux))
    
    def get_spectre(self, id):
        self.spectre.seek(id+1)
        spectre = self.spectre.readline()
        spectre = json.loads(spectre)
        return np.array(spectre["spectra"])

    def __data_generation(self, list_temp):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.array((self.batch_size/2, *self.dim))
        y = np.array((self.batch_size/2), dtype=int)

        # Generate data
        for i in range(0,len(list_temp),2):
            # Espectro del documento
            s1 = self.get_spectre(list_temp[i].idtext)
            s2 = self.get_spectre(list_temp[i+1].idtext)

            if self.op == 1: 
                X[i,] = self.test_1(s1, s2)
            else:
                X[i,] = self.test_2(s1, s2)
                if self.op == 3:
                    X[i,] = self.softmax(X[i,])

            # Verdad del conjunto de problemas
            y[i] = self.labels[list_temp[i].id]

        return X, keras.utils.to_categorical(y, num_classes=1)

class NeuralNetworkMLPNN:
    def __init__(self,train, truth_train, val, truth_val, op=1):
        self.trains = train
        self.truth_train = truth_train
        self.val = val
        self.truth_val = truth_val
        self.op = op
        self.size = (1200,) if op == 2 else (2400,)
        self.model = self.create_model()

    def train(self, epochs=100):
        # Generators
        training_generator = corpusGenerator(self.trains, self.truth_train, self.op, dim=self.size)
        validation_generator = corpusGenerator(self.val, self.truth_val, self.op, dim=self.size)
        # Train model on dataset with generator.
        self.model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6,epochs=epochs)

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers(input_shape=self.size),
            tf.keras.layers.Dense(600, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

        return model

def read_info(path):
    infile = open(path,'rb')
    file = pickle.load(infile)
    infile.close()
    return file

if __name__ == "__main__":
    pXid = './pan22-authorship-verification-training-dataset/particionesXid/'
    p = './pan22-authorship-verification-training-dataset/particiones/'
    test = NeuralNetworkMLPNN(read_info(pXid+"PanTrain.plk"), 
                              p+"train_truth.jsonl",
                              read_info(pXid+"PanVal.plk"), 
                              p+"val_truth.jsonl", op=2)
    test.train()