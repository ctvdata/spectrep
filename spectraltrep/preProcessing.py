from abc import ABCMeta
from abc import abstractmethod
from threading import Thread
import os
import pandas as pd
import nltk
import json
from featureExtraction import Writer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class Dispatcher(metaclass=ABCMeta):
    @abstractmethod
    def getBatch(self):
        pass

class CorpusReader(Dispatcher):
    def __init__(self, inputPath=None, batchSize=32):
        self.__inputPath = inputPath
        self.__batchSize = batchSize
        self.__idBatch = 0
        self.__number_of_lines = self.count_lines()

    def getBatch(self):
        # Si el número lineas procesadas es mayor o igual
        #  al número de líneas totales, solo mandamos el token.
        processed_lines = self.__idBatch * self.__batchSize #512
        if processed_lines >= self.__number_of_lines:
            return '<EOC>'

        # Aumentamos el id del batch.
        self.__idBatch += 1

        # Variable auxiliar que indica en que línea vamos.
        current_line = 1

        with open(self.__inputPath) as infile:
            # Variables auxiliares para el mínimo y máximo rango del batchSize.
            min_range = self.__batchSize * (self.__idBatch - 1)
            max_range = self.__batchSize * self.__idBatch

            # Lista de diccionarios que representará el batch de documentos.
            batch = []

            for line in infile:
                # Revisamos si estamos dentro del rango del batchSize.
                if current_line > min_range and current_line < max_range:
                    batch.append(json.loads(line))
                elif current_line == max_range:
                    batch.append(json.loads(line))
                    break
                
                current_line += 1
            
            # Regresamos el batch actual junto con su id.
            return self.__idBatch, batch
    
    # Método auxiliar para contar el número de líneas del archivo
    # Esto ayudará a que una vez que se termine de leer el archivo, no se
    # vuelva a leer el archivo.
    def count_lines(self):
        lines = 0

        with open(self.__inputPath) as infile:
            for line in infile:
                lines += 1

        return lines

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
        self.__wordnet_lemmatizer = WordNetLemmatizer()
    
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
        text = [self.__wordnet_lemmatizer.lemmatize(w) for w in text]
        
        text = ' '.join(text)    
        
        return text

    def run(self):
        while(True):
            batch = self.__dispatcher.getBatch()
            if(batch != '<EOC>'):
                documents = [self.__preProcess(t) for t in batch[1]]

                self.__sink.addPreprocessedBatch(self, (batch[0], documents))
            else:
                break

class SyntacticPreprocessor(Preprocessor):
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
        text = word_tokenize(text)
        text = nltk.pos_tag(text)
        text = ' '.join(text)    
        
        return text

    def run(self):
        while(True):
            batch = self.__dispatcher.getBatch()
            if(batch != '<EOC>'):
                documents = [self.__preProcess(t) for t in batch[1]]

                self.__sink.addPreprocessedBatch(self, (batch[0], documents))
            else:
                break

class SemanticPreprocessor(Preprocessor, Thread):
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
        
        return text

    def run(self):
        while(True):
            batch = self.__dispatcher.getBatch()
            if(batch != '<EOC>'):
                documents = [self.__preProcess(t) for t in batch[1]]

                self.__sink.addPreprocessedBatch(self, (batch[0], documents))
            else:
                break

class PreprocessorAbstractFactory(metaclass=ABCMeta):
    @abstractmethod
    def createLexicPreprocessor(self, dispatcher, sink):
        pass

    @abstractmethod
    def createSyntacticPreprocessor(self, dispatcher, sink):
        pass
    
    @abstractmethod
    def createSemanticPreprocessor(self, dispatcher, sink):
        pass

class PreprocessorFactory(PreprocessorAbstractFactory):
    def createLexicPreprocessor(self, dispatcher, sink):
        return LexicPreprocessor(dispatcher, sink)
    
    def createSyntacticPreprocessor(self, dispatcher, sink):
        return SyntacticPreprocessor(dispatcher, sink)

    def createSemanticPreprocessor(self, dispatcher, sink):
        return SemanticPreprocessor(dispatcher, sink)

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