from abc import ABCMeta
from abc import abstractmethod
from threading import Thread

class Document:     
    def __init__(self, text=None):
        self.__text = text
    
    # text getter method
    @property
    def text(self):
        return self.__text

    # text setter method
    @text.setter
    def text(self, text):
        self.__text = text

class Dispatcher(metaclass=ABCMeta):
    @abstractmethod
    def readCorpus(self):
        pass

    @abstractmethod
    def getBatch(self):
        pass

class CorpusReader(Dispatcher):
    def __init__(self, inputPath=None, batchSize=32):
        self.__inputPath = inputPath
        self.__batchSize = batchSize
        self.__corpus = self.__readCorpus()

    def __readCorpus(self):
        pass

    def getBatch(self):
        pass

class Preprocessor(metaclass=ABCMeta):
    @abstractmethod
    def __preProcess(self, batchId, documents):
        pass

class LexicPreprocessor(Preprocessor, Thread):
    def __init__(self):
        Thread.__init__(self)

    def __preProcess(self, batchId, documents):
        pass

    def run(self):
        pass

class SyntacticPreprocessor(Preprocessor):
    def __init__(self):
        Thread.__init__(self)

    def __preProcess(self, batchId, documents):
        pass

    def run(self):
        pass

class SemanticPreprocessor(Preprocessor, Thread):
    def __init__(self):
        Thread.__init__(self)

    def __preProcess(self, batchId, documents):
        pass

    def run(self):
        pass

class PreprocessorAbstractFactory(metaclass=ABCMeta):
    @abstractmethod
    def createLexicPreprocessor(self):
        pass

    @abstractmethod
    def createSyntacticPreprocessor(self):
        pass
    
    @abstractmethod
    def createSemanticPreprocessor(self):
        pass

class PreprocessorFactory(PreprocessorAbstractFactory):
    def createLexicPreprocessor(self):
        pass
    
    def createSyntacticPreprocessor(self):
        pass

    def createSemanticPreprocessor(self):
        pass

class Sink(metaclass=ABCMeta):
    @abstractmethod
    def addPreprocessedBatch(self, batchId, documents):
        pass
    
    @abstractmethod
    def __sortBatches(self):
        pass

    @abstractmethod
    def saveCorpus(self, outputPath):
        pass

class DocumentSink(Sink):
    __corpus = None
    
    def addPreprocessedBatch(self, batchId, documents):
        pass
    
    def __sortBatches(self):
        pass

    def saveCorpus(self, outputPath):
        pass