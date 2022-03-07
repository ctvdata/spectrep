from abc import ABCMeta, abstractmethod

class Reader(metaclass=ABCMeta):
    @abstractmethod
    def read_spectra(self, path):
        pass

class CorpusReader(Reader):
    def read_spectra(self, path):
        pass

class Resambler(metaclass=ABCMeta):
    @abstractmethod
    def resamble(self, spectra, metadata):
        pass

class Projector(Resambler):
    def resamble(self, spectra, metadata):
        pass