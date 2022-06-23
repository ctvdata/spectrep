import numpy as np
from abc import ABCMeta
from abc import abstractmethod
import json
from threading import Lock

class Sink(metaclass=ABCMeta):
    """
    Document sink.
    
    Used to collect processed documents.
    """
    @abstractmethod
    def addPreprocessedBatch(self, batch):
        """Adds a pre-processed batch of documents to a collection."""
        pass

    @abstractmethod
    def saveCorpus(self):
        """Saves the collection of processed documents to disk."""
        pass

class DocumentSink(Sink):
    """
    Document sink

    Used to collect processed documents.

    Args:
        sorted (bool): Indicates if the processed corpus will be saved in
        the same input order.
        
        outputPath (str): File output path.
    """ 
    
    def __init__(self, outputPath, sorted):
        self.__lock = Lock()
        self.__corpus = {}
        self.__outputPath = outputPath
        self.__sorted = sorted
        
    
    def addPreprocessedBatch(self, batch):
        """
        Adds a pre-processed batch of documents to a collection.

        Args:
            batch (dic): Diccionry with batch id and list of pre-processed documents.
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
        Sort the corpus by document id.
        """
        self.__corpus = {k: v for k, v in sorted(self.__corpus.items())}

    def saveCorpus(self):
        """
        Saves the pre-processed corpus to list.
        """
        self.__sortBatches() 
        write = "w" if self.__sorted else "a+"
        with open(self.__outputPath, write) as f:
            for batch in self.__corpus.values():
                for text in batch:
                    f.write(json.dumps(text, cls=_NumpyEncoder) + "\n")
    
class _NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
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
    Allows to read a corpus is batches.
    """
    @abstractmethod
    def getBatch(self):
        pass

class CorpusReader(Dispatcher):
    """
    Allows to read a corpus is batches.

    Attributes: 
        inputPath (str): File input path.
        batchSize (int): Size of batches.
    """
    def __init__(self, inputPath=None, batchSize=3000):
        """Initialized the CorpusReader."""
        self.__inputPath = inputPath
        self.__batchSize = batchSize

    def getBatch(self):
        """
        Creates document batches with an unique id.

        Yields:
        (int, list of dictionaries): Tuple with unique id and batch of documents.
        """
               
        batch = []
        processedLines = 0
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
    Used to ensure muti-threading sincronization.
    """
    def __init__(self, it):
        """Initializes the LockedIterator"""
        self.__lock = Lock()
        self.__it = iter(it)

    def __iter__(self):
        return self

    def __next__(self):
        with self.__lock:
            return next(self.__it, '<EOC>')