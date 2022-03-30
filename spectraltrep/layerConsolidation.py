from abc import ABCMeta, abstractmethod
import json
from threading import Thread, Lock
import numpy as np
from spectraltrep.preProcessing import DocumentSink

class Reader(metaclass=ABCMeta):
    @abstractmethod
    def read_spectre(self, path):
        pass

class CorpusReader(Reader):
    """
    Nos permite mandar el espectro (spectre) del Corpus línea por línea 
    para que no se sobrecargue la memoria.
    """

    def __init__(self):
        self.__lock = Lock()

    def read_spectre(self, path):
        """
        Generador que leé el archivo de tipo jsonl y regresa el id y su
        espectro correspondiente y así evitar sobrecargar la memoria.
        
        Args:
            path (str): La ruta del archivo jsonl que corresponde al spectre.
        
        Returns:
            (int, lista bidimensional de tipo double): El id y su spectre correspondiente.
        """
        with self.__lock:
            with open(path) as infile:
                for line in infile:
                    spectre_line = json.loads(line)
                    yield spectre_line['id'], spectre_line['spectre']

class Resambler(metaclass=ABCMeta):
    @abstractmethod
    def resamble(self, *spectra):
        pass

class Projector(Resambler):
    """
    Permite unificar los espectros de las diferentes características de un corpus

    Args:
        outputPath (str): Ruta del archivo de salida que contendrá los espectros unificados 
    """ 

    def __init__(self, outputPath):
        self.__outputPath = outputPath
        
    def __unify(self, spectra):
        """
        Junta los espectros de un texto en un solo vector
        
        Args:
            spectra (list): Tupla que contiene (id, espectros)
        
        Returns:
            (int, list): El id y una lista que contiene las listas de cada espectro.
        """
        id = spectra[0][0]
        vectors = np.array([v[1][1] for v in spectra])
  
        return id,[{'id': id, 'spectra': vectors}]

    def resamble(self, *spectra):
        """
        Unifica los espectros de un corpus y los guarda en un archivo

        Args:
            *spectra (str): Cada parámetro será la ruta del archivo donde 
            se encuentra la información de cada spectro
        """
        ds = DocumentSink(self.__outputPath, False)
        docReader = [CorpusReader() for i in spectra]
        generators = [doc.read_spectre(s) for doc, s in zip(docReader, spectra)]

        try:
            while True:
                batch = [next(gen) for gen in generators]
                ds.addPreprocessedBatch(self.__unify(batch))
        except StopIteration:
            print("Información guardada")

