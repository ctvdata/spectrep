from abc import ABCMeta
from abc import abstractmethod
from threading import Thread, Lock
import json
from xmlrpc.client import boolean
import nltk
import numpy as np
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
    def __init__(self, inputPath=None, batchSize=3000):
        self.__inputPath = inputPath
        self.__lock = Lock()
        self.__batchSize = batchSize
        self.__idBatch = 0
        self.__numberOfLines = self.__countLines()

    def getBatch(self):
        with self.__lock:
            # Si el número lineas procesadas es mayor o igual
            #  al número de líneas totales, solo mandamos el token.
            processedLines = self.__idBatch * self.__batchSize #512
            if processedLines >= self.__numberOfLines:
                return '<EOC>'

            # Aumentamos el id del batch.
            self.__idBatch += 1

            # Variable auxiliar que indica en que línea vamos.
            currentLine = 1

            with open(self.__inputPath) as infile:
                # Variables auxiliares para el mínimo y máximo rango del batchSize.
                minRange = self.__batchSize * (self.__idBatch - 1)
                maxRange = self.__batchSize * self.__idBatch

                # Lista de diccionarios que representará el batch de documentos.
                batch = []

                for line in infile:
                    # Revisamos si estamos dentro del rango del batchSize.
                    if currentLine > minRange and currentLine < maxRange:
                        batch.append(json.loads(line))
                    elif currentLine == maxRange:
                        batch.append(json.loads(line))
                        break
                    
                    currentLine += 1

                # Regresamos el batch actual junto con su id.
                return self.__idBatch, batch
    
    # Método auxiliar para contar el número de líneas del archivo
    # Esto ayudará a que una vez que se termine de leer el archivo, no se
    # vuelva a leer el archivo.
    def __countLines(self):
        lines = 0

        with open(self.__inputPath) as infile:
            for _ in infile:
                lines += 1

        return lines

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
    def saveCorpus(self):
        pass

class DocumentSink(Sink):
    """
    Permite guardar el corpus limpio en una archivo
    @type  order: bool
    @param order: Indica si el corpus se guardará ordenado
    @type  outputPath: str
    @param outputPath: Ruta del archivo de salida que contendrá el corpus
    """ 
    
    def __init__(self, order, outputPath):
        self.__lock = Lock()
        self.__corpus = {}
        self.__order = order
        self.__outputPath = outputPath
    
    def addPreprocessedBatch(self, batch):
        """
        Recibe un bloque del corpus, en caso de querer guardar el corpus
        ordenado los bloques se van almacenando, en caso contrario se guardan
        de manera inmediata en el archivo con la ruta de salida establecida.

        Método pensado para que varios hilos de ejecución puedan utilizarlo.
        @type  batch: dict
        @param batch: Diccionario con el número de bloque y el contenido del mismo
        """
        with self.__lock:
            self.__corpus[batch[0]] = batch[1]
            if not self.__order:
                #Guardamos en el archivo
                self.saveCorpus()
                #Limpiamos el corpus
                self.__corpus = {}
    
    def __sortBatches(self):
        """
        Ordena el corpus dado el id del bloque en el corpus original
        """
        self.__corpus = {k: v for k, v in sorted(self.__corpus.items())}

    def saveCorpus(self):
        """
        Guarda el corpus en la ruta establecida 
        """
        #no importa si no se quiere ordenar porque en ese caso sólo mandamos un bloque
        self.__sortBatches() 
        #Si se quiere ordenar sobreescribimos el archivo, en caso contrario iremos 
        # agregando los bloques sin eliminar los anteriores
        write = "w" if self.__order else "a+"
        with open(self.__outputPath, write) as f:
            for batch in self.__corpus.values():
                for text in batch:
                    #Formato de jsonl
                    f.write(json.dumps(text, cls=NumpyEncoder) + "\n")
    
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
