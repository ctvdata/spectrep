from abc import ABCMeta
from abc import abstractmethod

class Reader(metaclass=ABCMeta):
    @abstractmethod
    def readFeatureVectors(self, inputPath):
        pass

class CorpusReader(Reader):
    def readFeatureVectors(self, inputPath):
        pass

class Projector:
    def __init__(self):
        self.__somModel = None
    
    @property
    def somModel(self):
        return self.__somModel

    @somModel.setter
    def somModel(self, model):
        self.__somModel = model

    def saveSomModel(self, outputPath):
        pass

    def fit(self, featureVectors):
        pass

    def getProjection(self, featureVectors):
        pass