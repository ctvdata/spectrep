from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import sys
sys.path.append('..')
from simpsom import SOMNet

class Reader(metaclass=ABCMeta):
    @abstractmethod
    def readFeatureVectors(self, inputPath):
        pass

class CorpusReader(Reader):
    def readFeatureVectors(self, inputPath):
        pass

class Projector:
    def __init__(self, netLength=20, learningRate=0.01, epochs=1000):
        self.netLength=netLength
        self.learningRate=learningRate
        self.epochs=epochs

    def fit(self, featureVectors):
        self.net = SOMNet(self.netLength, self.netLength, featureVectors, PBC=True)
        self.net.train(self.learningRate, self.epochs)

    def getProjection(self, featureVectors):
        """
        X: Matrix to project over the som grid of size (num of vectors, num of features)
        nodeOperation: 'dot': Dot product or 'invEuc': inverse of euclidean distance
        """
        nrows = featureVectors.shape[0]
        i=0
        result_matrix = []
        for row in np.array(featureVectors): 
            j=0
            image_document=np.zeros(shape=self.netLength*self.netLength, dtype=np.float32)
            for node in self.net.nodeList:
                image_document[j] = 1/np.linalg.norm(row-node.weights)
                j=j+1
            mn=image_document.min()
            mx=image_document.max()
            image_document = (255.9 * (image_document-mn)/(mx-mn)).astype(np.uint8)
            image_document=image_document.reshape(self.netLength,self.netLength)
            result_matrix.append(image_document)  
            s = "{}% Complete".format(int((i*100)/nrows))
            print(s, end='\r')
            i=i+1
        print("100% Complete", end='\r')
        return result_matrix 

    def saveSomModel(self, outputPath):
        pass

    def loadSomModel(self, inputPath):
        pass