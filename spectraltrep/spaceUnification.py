from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import sys
sys.path.append('..')
from minisom.minisom import MiniSom
import json
import pickle

class Reader(metaclass=ABCMeta):
    """
    Feature vectors reader.
    """
    @abstractmethod
    def readFeatureVectors(self):
        """
        Loads feature vectors in memory.
        """
        pass

class VectorReader(Reader):
    """
    Feature vectors reader.

    Attributes:
        inputPath (str): Input file path. This file must be a valid jsonl
        file with the next format:

            {"id": document id, "vector": feature vector}
            ...
    """

    def __init__(self, inputPath):
        """
        Initializes the feature vector reader.
        """
        self.__inputPath = inputPath
        self.__numLines = 0

        with open(self.__inputPath) as f:
            for _ in f:
                self.__numLines +=1

    # def readFeatureVector(self):
    #     """
    #     Generador que carga vectores de características en memoria.
    #     Esta función no está en uso para esta versión de la api.

    #     Yields:
    #         Vector de características.
    #     """

    #     with open(self.__inputPath) as f:
    #             for line in f:
    #                 vector = json.loads(line)
    #                 yield vector

    def readFeatureVectors(self):
        """
        Loads feature vectors in memory.

        Returns:
            Feature vectors set.
        """
        vectors = list()
        with open(self.__inputPath) as f:
            for line in f:
                vector = json.loads(line)
                vectors.append(vector['vector'])
        return np.array(vectors)

class Projector:
    """
    Feature vector projector.
    
    Uses a Self-Organizing Maps (SOM) model to obtain content spectra.

    Attributes:
        netLength (int): Shape for the specter to obtain (netLength * netLength).

        numDimensions: Number of dimensions in the training vectors.

        learningRate: Learning rate for the SOM model.
    """

    def __init__(self, netLength, numDimensions, learningRate=0.01):
        """
        Initializes the vector projector.
        """

        self.__netLength=netLength
        self.__model = MiniSom(netLength, netLength, numDimensions, learning_rate=learningRate)

    def fit(self, featureVectors, epochs=10):
        """
        Starts the SOM model training.

        Args:
            featureVectors (numpy.ndarray): Set of training feature vectors.

            epochs (int): Number of training iterations.
        """

        self.__model.pca_weights_init(featureVectors)
        self.__model.train(featureVectors, epochs)

    def getProjection(self, featureVectors, documentSink=None):
        """
        Gets the feature vectors projection in a pretrained SOM model.

        Args:
            featureVectors (numpy.ndarray): Set of feature vectors to project.

            documentSink (DocumentSink): Projected vectors sink. When a
            documentSink is not provided a projection matrix is returned.

        Returns:
            Projection matrix in case a projected vectors sink is not 
            provided (DocumentSink).
        """
        nrows = featureVectors.shape[0]
        i=0
        result_matrix = []
        nodes = self.__model.get_weights()
        for row in featureVectors:
            image_document = np.zeros(shape=(self.__netLength,self.__netLength), dtype=np.float32)
        
            for i in np.arange(image_document.shape[0]):
                for j in np.arange(image_document.shape[1]):
                    image_document[i,j] = 1/np.linalg.norm(row-nodes[i,j])
            
            result_matrix.append(image_document)
            s = "{}% Complete".format(int((i*100)/nrows))
            print(s, end='\r')
            i=i+1
        
        result_matrix = np.array(result_matrix)
        result_matrix = 255.9 * (result_matrix - result_matrix.min()) / (result_matrix.max() - result_matrix.min())

        print("100% Complete", end='\r')
        
        if documentSink is None:
            return result_matrix
        else:
            spectra = []
            for idx, spectre in enumerate(result_matrix[:]):
                spectra.append({'id':idx, 'spectre':spectre})    
            documentSink.addPreprocessedBatch((0,spectra))

    def saveSomModel(self, outputPath):
        """
        Saves the trained SOM model.

        Args:
            outputPath (str): SOM model output path.
        """
        with open(outputPath, 'wb') as outfile:
            pickle.dump(self.__model, outfile)

    def loadSomModel(self, inputPath):
        """
        Loads a pretrained SOM model.

        Args: 
            inputPath (str): File input path.
        """
        with open(inputPath, 'rb') as infile:
            self.__model = pickle.load(infile)
            self.__netLength = self.__model.get_weights().shape[0]