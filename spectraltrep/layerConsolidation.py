from abc import ABCMeta, abstractmethod
import json
import numpy as np
from spectraltrep.utils import DocumentSink

class Reader(metaclass=ABCMeta):
    @abstractmethod
    def readSpectra(self):
        pass

class SpectraReader(Reader):
    """
    Nos permite mandar el espectro (spectre) del Corpus línea por línea 
    para que no se sobrecargue la memoria.
    
    Args:
        path (str): La ruta del archivo jsonl que corresponde al spectre.
    """

    def __init__(self, inputPath):
        self.__inputPath = inputPath

    def readSpectra(self):
        """
        Generador que leé el archivo de tipo jsonl y regresa el id y su
        espectro correspondiente y así evitar sobrecargar la memoria.
        
        Returns:
            (int, lista bidimensional de tipo double): El id y su spectre correspondiente.
        """
        with open(self.__inputPath) as infile:
            for line in infile:
                spectre_line = json.loads(line)
                yield spectre_line['id'], spectre_line['spectre']

class Assembler(metaclass=ABCMeta):
    @abstractmethod
    def assemble(self, *spectra):
        pass

class SpectraAssembler(Assembler):
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
        vectors = np.array([v[1] for v in spectra])

        return id,[{'id': id, 'spectra': vectors}]

    def assemble(self, *spectra):
        """
        Unifica los espectros de un corpus y los guarda en un archivo

        Args:
            *spectra (str): Cada parámetro será la ruta del archivo donde 
            se encuentra la información de cada spectro
        """
        ds = DocumentSink(self.__outputPath, False)
        docReader = [SpectraReader(i) for i in spectra]
        generators = [dr.readSpectra() for dr in docReader]

        try:
            while True:
                batch = [next(gen) for gen in generators]
                ds.addPreprocessedBatch(self.__unify(batch))
        except StopIteration:
            print("Información guardada")

