from abc import ABCMeta, abstractmethod

class Reader(metaclass=ABCMeta):
    """
    Interfaz para leer documentos
    """

    @abstractmethod
    def read_corpus(self) -> list:
        """
        
        :return: lista de documentos 
        """
        pass

class Vectorizer(metaclass=ABCMeta):
    """
    Interfaz para crear vectores de un documento
    según un enfoque léxico, sintáctico o semántico  
    """

    @abstractmethod
    def fit(self, corpus):
        """
        :param corpus: instancia de la clase Corpus 
        """
        pass

    @abstractmethod
    def transform(self, corpus):
        """
        :param corpus: instancia de la clase Corpus 
        :return: lista de vectores 
        """
        pass

class PreprocessorAbstractFactory(metaclass=ABCMeta):
    """
    Interfaz para crear procesadores con
    un enfoque léxico, sintáctico o semántico  
    """

    @abstractmethod
    def create_lexic_vectorizer(self):
        """
        :return: Un vector léxico
        """
        pass

    @abstractmethod
    def create_syntactic_vectorizer(self):
        """
        :return: Un vector sintáctico
        """
        pass

    @abstractmethod
    def create_semantic_vectorizer(self):
        """
        :return: Un vector semántico
        """
        pass

class Writer(metaclass=ABCMeta):
    """
    Interfaz para escribir los vectores obtenidos 
    del análisis léxico, sintáctico o semántico 
    """
    
    @abstractmethod
    def save_vectors(self, feature_vectors):
        """
        
        :param featureVectors: vectores a guardar  
        """
        pass

class Document():
    """
    Clase que modela un conjunto de documentos 
    """

    def __init__(self) -> None:
        super().__init__()
        self.__text = ''

    @property
    def text(self) -> str:
        """
        :return: Regresa el contenido del documento
        """
        return self.__text

    @text.setter
    def set_text(self, text) -> None:
        """
        :param text: texto del documento
        """
        self.__text = text

class Corpus():
    """
    Clase que modela un conjunto de documentos 
    """

    def __init__(self) -> None:
        super().__init__()
        self.__documents = []

    @property
    def documents(self) -> list(Document):
        """
        :return: Regresa la lista de documentos
        que conforman el Corpus
        """
        return self.__documents

    @documents.setter
    def set_input_path(self, documents) -> None:
        """
        :param document: lista de documentos
        que conforman el Corpus
        """
        self.__documents = documents

    def get_document(self, id) -> Document:
        """
        Regresa un documento del Corpus
        :param id: id - lugar del documento en la lista
        """
        return self.__documents[id]

    def add_document(self, document) -> None:
        """
        Agrega un documento al Corpus
        :param document: documento
        """
        self.__documents.append(document)

class CorpusReader(Reader):
    """
    Dada una ruta nos permite acceder al Corpus
    """

    def __init__(self) -> None:
        super().__init__()
        self.__input_path = ''

    @property
    def input_path(self) -> str:
        """
        :return: Regresa la ruta del Corpus
        """
        return self.__input_path

    @input_path.setter
    def set_input_path(self, path) -> None:
        """
        :param path: ruta del Corpus
        """
        self.__input_path = path

    def read_corpus(self) -> Corpus:
        """
        
        :return: regresa el Corpus (lista de documentos)
        """
        pass

class LexicVectorizer(Vectorizer):
    """
    Modela un vector para extaer características léxicas
    """

    def __init__(self) -> None:
        super().__init__()
        self.__model = ''

    @property
    def model(self) -> str:
        """
        :return: Regresa el modelo de los vectores
        """
        return self.__model

    @model.setter
    def set_model(self, model) -> None:
        """
        :param model: modelo de los vectores
        """
        self.__model = model
    
    def fit(self, corpus):
        """
        :param corpus: instancia de la clase Corpus 
        """
        pass

    def transform(self, corpus) -> list:
        """
        :param corpus: instancia de la clase Corpus 
        """
        return []

class SyntacticVectorizer(Vectorizer):
    """
    Modela un vector para extaer características sintácticas
    """

    def __init__(self) -> None:
        super().__init__()
        self.__model = ''

    @property
    def model(self) -> str:
        """
        :return: Regresa el modelo de los vectores
        """
        return self.__model

    @model.setter
    def set_model(self, model) -> None:
        """
        :param model: modelo de los vectores
        """
        self.__model = model
    
    def fit(self, corpus):
        """
        :param corpus: instancia de la clase Corpus 
        """
        pass

    def transform(self, corpus) -> list:
        """
        :param corpus: instancia de la clase Corpus 
        """
        return []

class SemanticVectorizer(Vectorizer):
    """
    Modela un vector para extaer características semánticas
    """

    def __init__(self) -> None:
        super().__init__()
        self.__model = ''

    @property
    def model(self) -> str:
        """
        :return: Regresa el modelo de los vectores
        """
        return self.__model

    @model.setter
    def set_model(self, model) -> None:
        """
        :param model: modelo de los vectores
        """
        self.__model = model
    
    def fit(self, corpus):
        """
        :param corpus: instancia de la clase Corpus 
        """
        pass
    
    def transform(self, corpus) -> list:
        """
        :param corpus: instancia de la clase Corpus 
        """
        return []

class VectorizerFactory(PreprocessorAbstractFactory):
    """
    Crear procesadores con un enfoque léxico, 
    sintáctico o semántico  
    """

    def create_lexic_vectorizer(self) -> LexicVectorizer:
        """
        :return: Un vector léxico
        """
        pass

    def create_syntactic_vectorizer(self) -> SyntacticVectorizer:
        """
        :return: Un vector sintáctico
        """
        pass

    def create_semantic_vectorizer(self) -> SemanticVectorizer:
        """
        :return: Un vector semántico
        """
        pass

class VectorWritter(Writer):
    """
    Guarda los vectores
    """

    def __init__(self) -> None:
        super().__init__()

    def save_vectors(self, feature_vectors) -> None:
        """
        
        :param feature_vectors: vectores a guardar  
        """
        pass
    
