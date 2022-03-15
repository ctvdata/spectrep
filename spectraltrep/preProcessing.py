from abc import ABCMeta
from abc import abstractmethod
from threading import Thread, Lock
import json
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  
import re
from nltk.stem import WordNetLemmatizer
wordnet.ensure_loaded()
import pdb

class Dispatcher(metaclass=ABCMeta):
    @abstractmethod
    def getBatch(self):
        pass

class CorpusReader(Dispatcher):
    """
    Nos permite mandar el Corpus en batches (lotes) para que no se
    sobrecargue la memoria.
    @type inputPath: str
    @param inputPath: La ruta del Corpus a leer.
    @type batchSize: int
    @param batchSize: Tamaño de los batches.
    """
    def __init__(self, inputPath=None, batchSize=3000):
        self.__inputPath = inputPath
        self.__lock = Lock()
        self.__batchSize = batchSize

    def getBatch(self):
        """
        Nos ayuda a leer el archivo del Corpus en formato jsonl
        para crear batches (lotes) de documentos a los cuáles se les asigna
        un id único.
        @rtype: (int, lista de diccionarios)
        @return: La tupla que contiene el id del batch y el batch actual.
        """
        with self.__lock:
            # Lista de diccionarios que representará el batch de documentos.
            batch = []
            # Líneas procesadas hasta el momento.
            processedLines = 0
            # Id del batch actual.
            idBatch = 0

            with open(self.__inputPath) as infile:
                for line in infile:
                    batch.append(json.loads(line))
                    processedLines += 1
                    
                    if processedLines == self.__batchSize:
                        idBatch += 1
                        processedLines = 0
                        yield idBatch, batch
                        batch = []
                if processedLines < self.__batchSize:
                    idBatch += 1
                    yield idBatch, batch

class Preprocessor(metaclass=ABCMeta):
    @abstractmethod
    def preProcess(self, text):
        pass

class LexicPreprocessor(Preprocessor, Thread):
    def __init__(self, dispatcher, sink):
        Thread.__init__(self)
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
    
    def preProcess(self, text):
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
                
                documents = []

                for t in batch[1]:
                    t['text'] = self.preProcess(t['text'])
                    documents.append(t)
                self.__sink.addPreprocessedBatch((batch[0], documents))
            else:
                break

class SyntacticPreprocessor(Preprocessor, Thread):
    def __init__(self, dispatcher, sink):
        Thread.__init__(self)
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

        self.__DELETE_NEW_LINE = re.compile('\\n') # Reemplazo de saltos de linea
        self.__DELETE_MIDSCORE = re.compile('-') # Eliminacion de guion medio
        self.__DELETE_PARENTHESES = re.compile('\(|\)') # Eliminación de paréntesis
        self.__DELETE_BRACKETS = re.compile('\[|\]') # Eliminacion de corchetes
        self.__REPLACE_DOUBLE_SPACE = re.compile('\s+') # Remplazo de espacios dobles
        self.__DELETE_QM = re.compile('"|’|\'') # Eliminar comillas
        self.__DELETE_PUNCTUATION = re.compile('[^\w\s]') # Eliminacion de signos de puntuacion
        self.__REPLACE_DIGITS = re.compile('\d') # Reemplazo de digitos
    
    def preProcess(self, text):
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
        text = ' '.join([t[1] for t in text])    
        
        return text

    def run(self):
        while(True):
            batch = self.__dispatcher.getBatch()
            if(batch != '<EOC>'):
                
                documents = []

                for t in batch[1]:
                    t['text'] = self.preProcess(t['text'])
                    documents.append(t)
                self.__sink.addPreprocessedBatch((batch[0], documents))
            else:
                break

class SemanticPreprocessor(Preprocessor, Thread):
    def __init__(self, dispatcher, sink):
        Thread.__init__(self)
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
            
        self.__DELETE_NEW_LINE = re.compile('\\n') # Reemplazo de saltos de linea
        self.__DELETE_MIDSCORE = re.compile('-') # Eliminacion de guion medio
        self.__DELETE_PARENTHESES = re.compile('\(|\)') # Eliminación de paréntesis
        self.__DELETE_BRACKETS = re.compile('\[|\]') # Eliminacion de corchetes
        self.__REPLACE_DOUBLE_SPACE = re.compile('\s+') # Remplazo de espacios dobles
        self.__DELETE_QM = re.compile('"|’|\'') # Eliminar comillas
        self.__DELETE_PUNCTUATION = re.compile('[^\w\s]') # Eliminacion de signos de puntuacion
        self.__REPLACE_DIGITS = re.compile('\d') # Reemplazo de digitos
    
    def preProcess(self, text):
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
                
                documents = []

                for t in batch[1]:
                    t['text'] = self.preProcess(t['text'])
                    documents.append(t)
                self.__sink.addPreprocessedBatch((batch[0], documents))
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
    def saveCorpus(self, outputPath):
        pass

class DocumentSink(Sink):
    def __init__(self):
        self.__lock = Lock()
        self.__corpus = {}


    
    def addPreprocessedBatch(self, batch):
        with self.__lock:
            self.__corpus[batch[0]] = batch[1]
    
    def __sortBatches(self):
        self.__corpus = {k: v for k, v in sorted(self.__corpus.items())}

    def saveCorpus(self, outputPath):
        self.__sortBatches() 

        with open(outputPath,"w") as f:
            for batch in self.__corpus.values():
                for  text in batch:
                    f.write("id: {0}, text: {1} \n".format(text['id'], text['text']))