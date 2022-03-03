from abc import ABCMeta, abstractmethod

class Reader(metaclass=ABCMeta):
    """
    Interfaz para leer matrices que contienen la información del 
    análisis léxico, sintáctico o semántico
    """

    @abstractmethod
    def read_spectra(self) -> list:
        """
        
        :return: lista con el espectro del documento
        """
        pass

class Resambler(metaclass=ABCMeta):
    """
    Interfaz para juntar los espectros 
    léxico, sintáctico o semántico 
    """

    @abstractmethod
    def resamble(self, spectra, metadata) -> None:
        """
        :param spectra: lista de espectros 
            léxico, sintáctico o semántico 
        :param metadata: Información adicional 
            (id del documento, etc)
        """
        pass

class CorpusReader(Reader):
    """
    Dada una ruta nos permite acceder al espectro del documento
    """

    def __init__(self) -> None:
        super().__init__()
        self.__input_path = ''

    @property
    def input_path(self) -> str:
        """
        :return: Regresa la ruta del espectro
        """
        return self.__input_path

    @input_path.setter
    def set_input_path(self, path) -> None:
        """
        :param path: ruta del espectro
        """
        self.__input_path = path

    def read_spectra(self) -> list:
        """
        
        :return: lista con el espectro del documento
        """
        pass

class Projector(Resambler):
    """
    Junta los espectros  léxico, sintáctico o semántico 
    """

    def __init__(self) -> None:
        super().__init__()

    def resamble(self, spectra, metadata) -> None:
        """
        :param spectra: lista de espectros 
            léxico, sintáctico o semántico 
        :param metadata: Información adicional 
            (id del documento, etc)
        """
        pass