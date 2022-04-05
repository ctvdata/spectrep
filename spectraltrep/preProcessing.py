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
from pathlib import Path

class Dispatcher(metaclass=ABCMeta):
    @abstractmethod
    def getBatch(self):
        pass

class CorpusReader(Dispatcher):
    """
    Nos permite mandar el Corpus en batches (lotes) para que no se
    sobrecargue la memoria.

    Args: 
        inputPath (str): La ruta del Corpus a leer

        batchSize (int): Tamaño de los batches
    """
    def __init__(self, inputPath=None, batchSize=3000):
        self.__inputPath = inputPath
        self.__batchSize = batchSize

    def getBatch(self):
        """
        Nos ayuda a leer el archivo del Corpus en formato jsonl
        para crear batches (lotes) de documentos a los cuáles se les asigna
        un id único.

        Returns:
        (int, lista de diccionarios): La tupla que contiene el id del batch y el batch actual.
        """
        
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
            if isinstance(dispatcher, LockedIterator):
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
            batch = next(self.__dispatcher)
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
            if isinstance(dispatcher, LockedIterator):
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
            batch = next(self.__dispatcher)
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
            if isinstance(dispatcher, LockedIterator):
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
        text = ' '.join(text)
        
        return text

    def run(self):
        while(True):
            batch = next(self.__dispatcher)
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

    Args:
        sorted (bool): Indica si el corpus se guardará ordenado
        
        outputPath (str): Ruta del archivo de salida que contendrá el corpus  
    """ 
    
    def __init__(self, outputPath, sorted):
        self.__lock = Lock()
        self.__corpus = {}
        self.__outputPath = outputPath
        self.__sorted = sorted
        
    
    def addPreprocessedBatch(self, batch):
        """
        Recibe un bloque del corpus, en caso de querer guardar el corpus
        ordenado los bloques se van almacenando, en caso contrario se guardan
        de manera inmediata en el archivo con la ruta de salida establecida.

        Método pensado para que varios hilos de ejecución puedan utilizarlo.

        Args:
            batch (dic): Diccionario con el número de bloque y el contenido del mismo
        """
        with self.__lock:
            self.__corpus[batch[0]] = batch[1]
            if not self.__sorted:
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
        write = "w" if self.__sorted else "a+"
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

class PreProcessingFacade():
    def preProcess(self, input, output, preProcessingType=["lex", "syn", "sem"], numThreads=1, batchSize=3000, sortedOutput=True):

        ppf = PreprocessorFactory()
        
        try:
            for tp in preProcessingType:
                cr = CorpusReader(input, batchSize).getBatch()
                lockedCr = LockedIterator(cr)
                p = Path(output)
                fileName = '{}_{}{}'.format(p.stem, tp, p.suffix)
                ds = DocumentSink(Path(p.parent, fileName), sortedOutput)

                # Inicializamos hilos de preprocesamiento
                PreprocessingThreads = []
                for _ in range(numThreads):
                    if tp=="lex":
                        pp = ppf.createLexicPreprocessor(lockedCr, ds)
                    elif tp=="syn":
                        pp = ppf.createSyntacticPreprocessor(lockedCr, ds)
                    elif tp=="sem":
                        pp = ppf.createSemanticPreprocessor(lockedCr, ds)
                    else:
                        raise Exception("Tipo de preprocesamiento no valido.")
                    
                    pp.start()
                    PreprocessingThreads.append(pp)

                # Esperamos a que terminen los hilos de preprocesamiento
                for t in PreprocessingThreads:
                    t.join()

                # Guardamos el corpus preprocesado en caso de que sea un corpus ordenado
                if sortedOutput:
                    ds.saveCorpus()

        except Exception as err:
            print(err)
        
class LockedIterator(object):
    def __init__(self, it):
        self.__lock = Lock()
        self.__it = iter(it)

    def __iter__(self):
        return self

    def __next__(self):
        with self.__lock:
            return next(self.__it, '<EOC>')