from abc import ABCMeta
from abc import abstractmethod

class Preprocessor(metaclass=ABCMeta):
    @abstractmethod
    def preProcess(self, documents):
        pass

class LexicPreprocessor(Preprocessor):
    def __init__(self):
        self.__threadId = -1

    # threadId getter method
    @property
    def threadId(self):
        return self.__threadId

    # threadId setter method
    @threadId.setter
    def threadId(self, id):
        self.__threadId = id

    def preProcess(self,documents):
        pass

class SemanticPreprocessor(Preprocessor):
    def __init__(self):
        self.__threadId = -1

    # threadId getter method
    @property
    def threadId(self):
        return self.__threadId

    # threadId setter method
    @threadId.setter
    def threadId(self, id):
        self.__threadId = id

    def preProcess(self,documents):
        pass

class SyntacticPreprocessor(Preprocessor):
    def __init__(self):
        self.__threadId = -1

    # threadId getter method
    @property
    def threadId(self):
        return self.__threadId

    # threadId setter method
    @threadId.setter
    def threadId(self, id):
        self.__threadId = id

    def preProcess(self,documents):
        pass

class PreprocessorAbstractFactory(metaclass=ABCMeta):
    def createLexicPreprocessor(threadId):
        pass
    
    def createSyntacticPreprocessor(threadId):
        pass

    def createSemanticPreprocessor(threadId):
        pass

class PreprocessorFactory(PreprocessorAbstractFactory):
    def __init__(self):
        super().__init__()

    def createLexicPreprocessor(threadId):
        pass
    
    def createSyntacticPreprocessor(threadId):
        pass

    def createSemanticPreprocessor(threadId):
        pass

class Dispatcher(metaclass=ABCMeta):
    @abstractmethod
    def getBatch(self):
        pass

class CorpusReader(Dispatcher):
    def __init__(self):
        self.__inputPath = ''
        self.__batchSize = 0
    
    # inputPath getter method
    @property
    def inputPath(self):
        return self.__inputPath

    # inputPath setter method
    @inputPath.setter
    def inputPath(self, path):
        self.__inputPath = path

    # batchSize getter method
    @property
    def batchSize(self):
        return self.__batchSize

    # batchSize setter method
    @batchSize.setter
    def batchSize(self, size):
        self.__batchSize = size

    # get Batch method
    def getBatch(self):
        pass

class Sink(metaclass=ABCMeta):
    @abstractmethod
    def addPreprocessedBatch(self, batchId, documents):
        pass
    
    @abstractmethod
    def sortBatches(self):
        pass

    @abstractmethod
    def saveCorpus(self):
        pass

class DocumentSink(Sink):
    def __init__(self):
        self.__corpus = None
    
    # corpus getter method
    @property
    def corpus(self):
        return self.__corpus

    # corpus setter method
    @corpus.setter
    def corpus(self, corpus):
        self.__corpus = corpus

    def addPreprocessedBatch(batchId, documents):
        pass

    def saveCorpus(self):
        pass

    def sortBatches(self):
        pass

class Thread(metaclass=ABCMeta):
    @abstractmethod
    def run(self):
        pass

class Document:
    def __init__(self):
        self.__text = ''
    
    # text getter method
    @property
    def text(self):
        return self.__text

    # text setter method
    @text.setter
    def text(self, val):
        self.__text = val
