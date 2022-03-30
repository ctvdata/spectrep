from abc import ABCMeta, abstractmethod
import json
from threading import Thread, Lock

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
        @type path: string
        @param path: La ruta del archivo jsonl que corresponde al spectre.
        @rtype: (int, lista bidimensional de tipo double)
        @return: El id y su spectre correspondiente.
        """
        with self.__lock:
            with open(path) as infile:
                for line in infile:
                    spectre_line = json.loads(line)
                    yield spectre_line['id'], spectre_line['spectre']

class Resambler(metaclass=ABCMeta):
    @abstractmethod
    def resamble(self, spectra, metadata):
        pass

class Projector(Resambler):
    def resamble(self, spectra, metadata):
        pass