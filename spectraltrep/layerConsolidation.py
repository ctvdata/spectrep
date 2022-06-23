from abc import ABCMeta, abstractmethod
import json
import numpy as np
from spectraltrep.utils import DocumentSink

class Reader(metaclass=ABCMeta):
    """Content spectra reader."""
    @abstractmethod
    def readSpectra(self):
        """Reads content spectra from an input file."""
        pass

class SpectraReader(Reader):
    """
    Content spectra reader.
    
    Attributes:
        inputPath (str): File input path. The input must be a valid jsonl
        file with the next format:

            {"id": document id, "spectre": document spectre}
            ...
    """

    def __init__(self, inputPath):
        """Initialized the spectra reader."""

        self.__inputPath = inputPath

    def readSpectra(self):
        """
        Reads a content spectre from the input file.
        
        Yields:
            (int, bidimensional list): Content spectre with its corresponding id.
        """
        with open(self.__inputPath) as infile:
            for line in infile:
                spectre_line = json.loads(line)
                yield spectre_line['id'], spectre_line['spectre']

class Assembler(metaclass=ABCMeta):
    """Content spectra assembler."""

    @abstractmethod
    def assemble(self, *spectra):
        """
        Assembles each content spectra from a document into a single matrix.
        """
        pass

class SpectraAssembler(Assembler):
    """
    Content spectra assembler.
    
    Unifies different content spectra of a single corpus.

    Attributes:
        outputPath (str): Output file for the assembled spectra.
    """ 

    def __init__(self, outputPath):
        """Initializes the spectra assembler."""
        self.__outputPath = outputPath
        
    def __unify(self, spectra):
        """
        Unifies all the spectra from a single document.
        
        Args:
            spectra (list): Tuple of the dorm (id, specter).
        
        Returns:
            (int, list): Id and all spectra from a single document 
            into a single matrix.
        """
        id = spectra[0][0]
        vectors = np.array([v[1] for v in spectra])

        return id,[{'id': id, 'spectra': vectors}]

    def assemble(self, *spectra):
        """
        Assembles each content spectra from a document into a single matrix.

        Args:
            List(str): List of spectra input paths.
        """
        ds = DocumentSink(self.__outputPath, False)
        docReader = [SpectraReader(i) for i in spectra]
        generators = [dr.readSpectra() for dr in docReader]

        try:
            while True:
                batch = [next(gen) for gen in generators]
                ds.addPreprocessedBatch(self.__unify(batch))
        except StopIteration:
            print("Information saved.")