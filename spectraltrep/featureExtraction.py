from abc import ABCMeta, abstractmethod
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class Reader(metaclass=ABCMeta):
    @abstractmethod
    def readCorpus(self, inputPath):
        pass

class CorpusReader(Reader):
    def readCorpus(self, inputPath):
        pass

class Corpus():
    @property
    def documents(self):
        return self.__documents

    @documents.setter
    def documents(self, documents):
        self.__documents = documents
    
    def addDocument(self, document):
        pass

    def getDocument(self):
        pass

class Document():
    def __init__(self, text=None):
        self.__text = text

    @property
    def text(self):
        return self.__text

    @text.setter
    def set_text(self, text):
        self.__text = text

class Vectorizer(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, corpus):
        pass

    @abstractmethod
    def transform(self, corpus):
        pass

    @abstractmethod
    def saveModel(self, outputPath):
        pass

class LexicVectorizer(Vectorizer):
    @property
    def model(self):
        return self.__model

    @model.setter
    def set_model(self, inputPath):
        self.__model = Doc2Vec().load(inputPath)
    
    def fit(self, corpus):
        pass

    def transform(self, corpus):
        pass

    def saveModel(self, outputPath):
        self.__model.save(outputPath)

class SyntacticVectorizer(Vectorizer):
    @property
    def model(self):
        return self.__model

    @model.setter
    def set_model(self, inputPath):
        self.__model = Doc2Vec().load(inputPath)
    
    def fit(self, corpus):
        pass

    def transform(self, corpus):
        pass

    def saveModel(self, outputPath):
        self.__model.save(outputPath)

class SemanticVectorizer(Vectorizer):
    def __init__(self, corpusReader, vectorWriter, vectorSize=50, minCount=1, epochs=10):
        self.__model = Doc2Vec(vector_size=vectorSize, min_count=minCount, epochs=epochs)
        self.__corpusReader = corpusReader
        self.__vectorWriter = vectorWriter

    @property
    def model(self):
        return self.__model

    @model.setter
    def set_model(self, inputPath):
        self.__model = Doc2Vec().load(inputPath)
    
    def fit(self):
        self.__model.build_vocab(self.__corpusReader)
        self.__model.train(self.__corpusReader, total_examples=self.__model.corpus_count, epochs=self.__model.epochs)
        pass

    def transform(self, corpus):
        pass

    def saveModel(self, outputPath):
        self.__model.save(outputPath)

class VectorizerAbstractFactory(metaclass=ABCMeta):
    @abstractmethod
    def create_lexic_vectorizer(self):
        pass

    @abstractmethod
    def create_syntactic_vectorizer(self):
        pass

    @abstractmethod
    def create_semantic_vectorizer(self):
        pass

class VectorizerFactory(VectorizerAbstractFactory):
    def create_lexic_vectorizer(self):
        pass

    def create_syntactic_vectorizer(self):
        pass

    def create_semantic_vectorizer(self):
        pass

class Writer(metaclass=ABCMeta):
    @abstractmethod
    def save_vectors(self, feature_vectors, path):
        pass

class VectorWritter(Writer):
    def save_vectors(self, feature_vectors, path):
        pass