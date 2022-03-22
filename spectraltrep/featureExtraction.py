from abc import ABCMeta, abstractmethod
import string
from tkinter import SE
from tokenize import String
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json
from nltk.tokenize import word_tokenize
import pdb

class Reader(metaclass=ABCMeta):
    @abstractmethod
    def readCorpus(self, inputPath):
        pass

class Doc2VecCorpusReader(Reader):
    def __init__(self, inputPath):
            self.__inputPath = inputPath

    def __iter__(self):        
        with open(self.__inputPath) as f:
            for line in f:
                line = json.loads(line)
                tokens = word_tokenize(line['text'])
                yield TaggedDocument(tokens, [int(line['id'])])

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
    def fit(self, corpus):
        pass

    def transform(self, corpus):
        pass

    def saveModel(self, outputPath):
        self.__model.save(outputPath)

class SyntacticVectorizer(Vectorizer):
    def fit(self, corpus):
        pass

    def transform(self, corpus):
        pass

    def saveModel(self, outputPath):
        pass

class SemanticVectorizer(Vectorizer):
    def __init__(self, vectorWriter, corpusReader=None, vectorSize=50, minCount=1, epochs=10):
        self.__model = Doc2Vec(vector_size=vectorSize, min_count=minCount, epochs=epochs)
        self.__vectorWriter = vectorWriter
        self.__corpusReader = corpusReader

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, inputPath):
        self.__model = Doc2Vec.load(inputPath)
    
    def fit(self):
        try:
            if self.__corpusReader is None:
                raise Exception('No se ha definido un corpus reader en el constructor para el entrenamiento.')
            else:
                self.__model.build_vocab(self.__corpusReader)
                self.__model.train(self.__corpusReader, total_examples=self.__model.corpus_count, epochs=self.__model.epochs)            
        except Exception as err:
            print(err)        

    def transform(self, corpusReader=None):
        if corpusReader is None:
            for doc in self.__corpusReader:
                idDoc = doc[1][0]
                docVec = self.__model.dv[idDoc]
                batch = (idDoc,[{'id':idDoc, 'vector':docVec}])
                self.__vectorWriter.addPreprocessedBatch(batch)
        else:
            for doc in corpusReader:
                idDoc = doc[1][0]
                tokens = doc[0]
                docVec = self.__model.infer_vector(tokens)
                batch = (idDoc,[{'id':idDoc, 'vector':docVec}])
                self.__vectorWriter.addPreprocessedBatch(batch)
        

    def saveModel(self, outputPath):
        self.__model.save(outputPath)

class VectorizerAbstractFactory(metaclass=ABCMeta):
    @abstractmethod
    def createLexicVectorizer(self):
        pass

    @abstractmethod
    def createSyntacticVectorizer(self):
        pass

    @abstractmethod
    def createSemanticVectorizer(self):
        pass

class VectorizerFactory(VectorizerAbstractFactory):
    def createLexicVectorizer(self):
        pass

    def createSyntacticVectorizer(self):
        pass

        
    def createSemanticVectorizer(self, vectorWriter, corpusReader=None, vectorSize=50, minCount=1, epochs=10):
        return SemanticVectorizer(vectorWriter, corpusReader, vectorSize, minCount, epochs)

class Writer(metaclass=ABCMeta):
    @abstractmethod
    def save_vectors(self, feature_vectors, path):
        pass

class VectorWritter(Writer):
    def save_vectors(self, feature_vectors, path):
        pass