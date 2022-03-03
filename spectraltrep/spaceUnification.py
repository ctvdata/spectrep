from abc import ABCMeta
from abc import abstractmethod

class Reader(metaclass=ABCMeta):
    @abstractmethod
    def readFeatureVectors(self):
        pass

class CorpusReader(Reader):
    def __init__(self):
        self.__inputPath = ''

    # inputPath getter method
    @property
    def inputPath(self):
        return self.__inputPath

    # inputPath setter method
    @inputPath.setter
    def inputPath(self, path):
        self.__inputPath = path

    def readFeatureVectors(self):
        pass

class Projector:
    def __init__(self):
        self.__somModel = None
    
    # somModel getter method
    @property
    def somModel(self):
        return self.__somModel

    # somModel setter method
    @somModel.setter
    def somModel(self, model):
        self.__somModel = model

    def saveSomModel():
        pass

    def fit(featureVectors):
        pass

    def getProjection(featureVector):
        pass
    