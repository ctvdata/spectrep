import numpy as np
from abc import ABCMeta
from abc import abstractmethod
import json
from threading import Lock

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
        Agrega un bloque preprocesado del corpus.
        
        En caso de querer guardar el corpus
        ordenado los bloques se van almacenando, en caso contrario se guardan
        de manera inmediata en el archivo con la ruta de salida establecida.
        Método pensado para que varios hilos de ejecución puedan utilizarlo.

        Args:
            batch (dic): Diccionario con el identificador de bloque y el contenido del mismo
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
        Guarda el corpus preprocesado en la ruta establecida 
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
                    f.write(json.dumps(text, cls=_NumpyEncoder) + "\n")
    
class _NumpyEncoder(json.JSONEncoder):
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

class Dispatcher(metaclass=ABCMeta):
    """
    Permite leer un corpus por lotes.
    """
    @abstractmethod
    def getBatch(self):
        pass

class CorpusReader(Dispatcher):
    """
    Permite leer un corpus por lotes.

    Attributes: 
        inputPath (str): La ruta del Corpus a leer.
        batchSize (int): Tamaño de los lotes.
    """
    def __init__(self, inputPath=None, batchSize=3000):
        """Inicializa CorpusReader"""
        self.__inputPath = inputPath
        self.__batchSize = batchSize

    def getBatch(self):
        """
        Crear lotes de documentos con un id único.

        Yields:
        (int, lista de diccionarios): La tupla que contiene el id del lote
        y el contenido del mismo.
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

class LockedIterator(object):
    """
    Se utiliza para asegurar la sicronización de múltiples hilos al leer un corpus.
    """
    def __init__(self, it):
        """Inicializa Locked Iterator"""
        self.__lock = Lock()
        self.__it = iter(it)

    def __iter__(self):
        return self

    def __next__(self):
        with self.__lock:
            return next(self.__it, '<EOC>')