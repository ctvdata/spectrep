import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, instances, labels, spectra, batch_size=32, dim=(1200,),
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.instances = instances
        self.spectra = spectra
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __get_pair(self, id):

        pair_spectra = self.instances.loc[self.instances.id == id].merge(self.spectra, left_on='idtext', right_on='id').spectra
        # x1 = pair_spectra[0].flatten()
        # x2 = pair_spectra[1].flatten()
        x1 = pair_spectra[0][2].flatten() # Modificado para traer solamente capa sintactica y semantica
        x2 = pair_spectra[1][2].flatten() # Modificado para traer solamente capa sintactica y semantica
        
        return x1, x2 
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X1 = np.empty((self.batch_size, *self.dim))
        X2 = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            pair = self.__get_pair(ID)
            X1[i,] = pair[0]
            X2[i,] = pair[1]

            # Store class
            y[i] = self.labels[ID]

        return (X1,X2), y

class TestGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, instances, spectra, batch_size=32, dim=(1200,)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.instances = instances
        self.spectra = spectra
        self.indexes = np.arange(len(self.list_IDs))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def __get_pair(self, id):

        pair_spectra = self.instances.loc[self.instances.id == id].merge(self.spectra, left_on='idtext', right_on='id').spectra
        # x1 = pair_spectra[0].flatten()
        # x2 = pair_spectra[1].flatten()
        x1 = pair_spectra[0][2].flatten() # Modificado para traer solamente capa sintactica y semantica
        x2 = pair_spectra[1][2].flatten() # Modificado para traer solamente capa sintactica y semantica
        
        return x1, x2 
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X1 = np.empty((self.batch_size, *self.dim))
        X2 = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            pair = self.__get_pair(ID)
            X1[i,] = pair[0]
            X2[i,] = pair[1]

        return (X1,X2),