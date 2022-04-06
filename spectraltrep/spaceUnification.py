from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import sys
sys.path.append('..')
from minisom.minisom import MiniSom
import json
import pickle

class Reader(metaclass=ABCMeta):
    @abstractmethod
    def readFeatureVectors(self):
        pass

class VectorReader(Reader):
    def __init__(self, inputPath):
        self.__inputPath = inputPath
        self.__numLines = 0

        with open(self.__inputPath) as f:
            for _ in f:
                self.__numLines +=1

    def readFeatureVector(self):
        with open(self.__inputPath) as f:
                for line in f:
                    vector = json.loads(line)
                    yield vector

    #Hay que buscar si es posibe acopar esta funcion a SimpSom
    # def readTrainingFeatureVectors(self, size):
    #     indices = np.random.randint(0, self.__numLines, size)
    #     for idx in indices:
    #         with open(self.__inputPath) as f:
    #             for line in f:
    #                 vector = json.loads(line)
    #                 if vector['id'] == idx:
    #                     yield np.array(vector['vector'])
    #                     break

    def readFeatureVectors(self):
        vectors = list()
        with open(self.__inputPath) as f:
            for line in f:
                vector = json.loads(line)
                vectors.append(vector['vector'])
        return np.array(vectors)

class Projector:
    def __init__(self, netLength, numDimensions, learningRate=0.01):
        self.__netLength=netLength
        self.__model = MiniSom(netLength, netLength, numDimensions, learning_rate=learningRate)

    def fit(self, featureVectors, epochs=10):
        self.__model.pca_weights_init(featureVectors)
        self.__model.train(featureVectors, epochs)

    def getProjection(self, featureVectors, documentSink=None):
        """
        X: Matrix to project over the som grid of size (num of vectors, num of features)
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
            


        # for row in featureVectors:
        #     j=0
        #     image_document=np.zeros(shape=self.__netLength*self.__netLength, dtype=np.float32)
        #     for node in self.net.nodeList:
        #         image_document[j] = 1/np.linalg.norm(row-node.weights)
        #         j=j+1
        #     mn=image_document.min()
        #     mx=image_document.max()
        #     image_document = (255.9 * (image_document-mn)/(mx-mn)).astype(np.uint8)
        #     image_document=image_document.reshape(self.netLength,self.netLength)
        #     result_matrix.append(image_document)  
        #     s = "{}% Complete".format(int((i*100)/nrows))
        #     print(s, end='\r')
        #     i=i+1
        # print("100% Complete", end='\r')
        # return result_matrix 

    def saveSomModel(self, outputPath):
        with open(outputPath, 'wb') as outfile:
            pickle.dump(self.__model, outfile)

    def loadSomModel(self, inputPath):
        with open(inputPath, 'rb') as infile:
            self.__model = pickle.load(infile)
            self.__netLength = self.__model.get_weights().shape[0]