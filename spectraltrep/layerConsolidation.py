from abc import ABCMeta, abstractmethod
import json
import numpy as np
from spectraltrep.utils import DocumentSink

class Reader(metaclass=ABCMeta):
    """Lector de espectros de contenido."""
    @abstractmethod
    def readSpectra(self):
        """Lee un espectro de contenido del archivo de entrada."""
        pass

class SpectraReader(Reader):
    """
    Lector de espectros de contenido.
    
    Permite cargar el espectro del Corpus línea por línea.
    
    Attributes:
        inputPath (str): Ruta del archivo de entrada. Este debe ser un archivo jsonl
            con el siguiente formato:

            {"id": idDelDocumento, "spectre": espectroDelDocumento}
            ...
    """

    def __init__(self, inputPath):
        """
        Inicialzia el lector de espectros.

        Args:
            inputPath (str): Ruta del archivo de entrada. Este debe ser un archivo jsonl
                con el siguiente formato:

                {"id": idDelDocumento, "spectre": espectroDelDocumento}
                ...
        """
        self.__inputPath = inputPath

    def readSpectra(self):
        """
        Lee un espectro de contenido del archivo de entrada.
        
        Yields:
            (int, lista bidimensional de tipo double): El id y su spectro correspondiente.
        """
        with open(self.__inputPath) as infile:
            for line in infile:
                spectre_line = json.loads(line)
                yield spectre_line['id'], spectre_line['spectre']

class Assembler(metaclass=ABCMeta):
    """Ensamblador de espectros de contenido"""

    @abstractmethod
    def assemble(self, *spectra):
        """
        Ensambla cada uno de los espectros de contenido en una sola matriz por documento.
        """
        pass

class SpectraAssembler(Assembler):
    """
    Ensamblador de espectros de contenido. 
    
    Permite unificar los espectros de las diferentes características de un corpus.

    Attributes:
        outputPath (str): Ruta del archivo de salida que contendrá los espectros ensamblados.
    """ 

    def __init__(self, outputPath):
        """
        Inicializa el ensamblador de espectros.

        Args:
            outputPath (str): Ruta del archivo de salida que contendrá
                los espectros ensamblados.
        """
        self.__outputPath = outputPath
        
    def __unify(self, spectra):
        """
        Unifica los diferentes espectros de un mismo documento.
        
        Args:
            spectra (list): Tupla que contiene (id, espectros).
        
        Returns:
            (int, list): El id y una lista que contiene los espectros de un mismo documento.
        """
        id = spectra[0][0]
        vectors = np.array([v[1] for v in spectra])

        return id,[{'id': id, 'spectra': vectors}]

    def assemble(self, *spectra):
        """
        Ensambla cada uno de los espectros de contenido en una sola matriz por documento.

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