from abc import ABCMeta
from abc import abstractmethod
from threading import Thread
import os
import pandas as pd
import nltk
from spectraltrep.featureExtraction import Writer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

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
    def __preProcess(self, text):
        pass

class LexicPreprocessor(Preprocessor, Thread):
    def __init__(self, dispatcher, sink):
        try:
            if isinstance(dispatcher, Dispatcher):
                self.__dispatcher = dispatcher
            else:
                raise Exception("Non-valid instance of dispatcher.")

            if isinstance(sink, Sink):
                self.__sink = sink
            else:
                raise Exception("Non-valid instance of Sink.")
        
        except Exception as err:
            print(err)

        Thread.__init__(self)        
        self.__DELETE_NEW_LINE = re.compile('\\n') # Reemplazo de saltos de linea
        self.__DELETE_MIDSCORE = re.compile('-') # Eliminacion de guion medio
        self.__DELETE_PARENTHESES = re.compile('\(|\)') # Eliminación de paréntesis
        self.__DELETE_BRACKETS = re.compile('\[|\]') # Eliminacion de corchetes
        self.__REPLACE_DOUBLE_SPACE = re.compile('\s+') # Remplazo de espacios dobles
        self.__DELETE_QM = re.compile('"|’|\'') # Eliminar comillas
        self.__DELETE_PUNCTUATION = re.compile('[^\w\s]') # Eliminacion de signos de puntuacion
        self.__REPLACE_DIGITS = re.compile('\d') # Reemplazo de digitos
        self.__stop_words = set(stopwords.words('english'))
    
    def __preProcess(self, text):
        text = text.lower()
        text = self.__DELETE_NEW_LINE.sub("", text)
        text = self.__DELETE_MIDSCORE.sub(" ", text)
        text = self.__DELETE_PARENTHESES.sub("", text)    
        text = self.__DELETE_BRACKETS.sub("", text)    
        text = self.__REPLACE_DOUBLE_SPACE.sub(" ", text)   
        text = self.__DELETE_QM.sub("", text)
        text = self.__DELETE_PUNCTUATION.sub('', text)
        text = self.__REPLACE_DIGITS.sub('<NUM>', text)
        
        word_tokens = word_tokenize(text)
        text = [w for w in word_tokens if not w in self.__stop_words]
        
        wordnet_lemmatizer = WordNetLemmatizer()    
        text = [wordnet_lemmatizer.lemmatize(w) for w in text]
        
        text = ' '.join(text)    
        
        return(text)

    def run(self):
        while(True):
            batch = self.__dispatcher.getBatch()
            if(batch != '<EOC>'):
                batchId = batch[0]
                documents = batch[1]
                documents = [self.__preProcess(t) for t in documents]

                self.__sink.addPreprocessedBatch(self, (batchId, documents))
            else:
                break

class SyntacticPreprocessor(Preprocessor):
    def __init__(self):
        Thread.__init__(self)

    def __preProcess(self, text):
        pass

    def run(self):
        pass

class SemanticPreprocessor(Preprocessor, Thread):
    def __init__(self):
        Thread.__init__(self)

    def __preProcess(self, text):
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
    def addPreprocessedBatch(self, batch):
        pass
    
    @abstractmethod
    def __sortBatches(self):
        pass

    @abstractmethod
    def saveCorpus(self, outputPath):
        pass

class DocumentSink(Sink):
    __corpus = None
    
    def addPreprocessedBatch(self, batch):
        pass
    
    def __sortBatches(self):
        pass

    def saveCorpus(self, outputPath):
        pass