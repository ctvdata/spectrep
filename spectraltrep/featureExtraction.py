from abc import ABCMeta, abstractmethod
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json
from nltk.tokenize import word_tokenize
import json
import numpy as np

class Doc2VecCorpusReader():
    """
    Generador para leer documentos de un corpus.

    Este gnerador es utilizado por la clase Doc2Vec del paquete GenSim

    Yields:
        TaggedDocument
    """

    def __init__(self, inputPath):
            self.__inputPath = inputPath

    def __iter__(self):        
        with open(self.__inputPath) as f:
            for line in f:
                line = json.loads(line)
                tokens = word_tokenize(line['text'])
                yield TaggedDocument(tokens, [int(line['id'])])

class Vectorizer(metaclass=ABCMeta):
    """Vectorizador de documentos de texto"""

    @abstractmethod
    def fit(self):
        """
        Entrena un modelo de vectorización de documentos.
        """
        pass

    @abstractmethod
    def transform(self, corpus):
        """
        Tansforma un corpus en un conjunto de vectores de características.
        """
        pass

    @abstractmethod
    def saveModel(self, outputPath):
        """Guarda el modelo de vectorización de documentos entrenado."""
        pass

class LexicVectorizer(Vectorizer):
    """
    Vectorizador léxico de documentos de texto.
    
    Attributes:
        vectorWriter (DocumentSink): Recolector de documentos vectorizados.
        corpusReader (CorpusReader): Lector de documentos de texto. Solamente se
        utiliza en la etapa de entrenamiento del modelo de vectorización.
    """

    def __init__(self, vectorWriter, corpusReader=None):
        """Inicializa el vectorizador léxico"""

        self.__model = LexicModel()
        self.__vectorWriter = vectorWriter
        self.__corpusReader = corpusReader

    @property
    def model(self):
        """Modelo de vectorización."""
        return self.__model

    @model.setter
    def model(self, inputPath):
        """
        Estabece un modelo de vectorización pre-entrnado.
        
        Args:
            inputPath (str): Ruta del modelo a cargar.
        """
        model = LexicModel()
        model.load(inputPath)
        self.__model = model

    def fit(self):
        """
        Entrena un modelo de vectorización léxico de documentos.
        """

        try:
            if self.__corpusReader is None:
                raise Exception('No se ha definido un corpus reader en el constructor para el entrenamiento.')
            else:
                gen = self.__corpusReader.getBatch()
                for batch in gen:
                    for doc in batch[1]:
                        print(f"Entrenando con doc {doc['id']}", end='\r')
                        doc = word_tokenize(doc['text'])
                        for token in doc:
                            self.__model.addToken(token)
                self.__model.setToken2Id()

        except Exception as err:
            print(err)

    def __getVector(self, text):
        # Creamos el vector de ceros
        vector = np.zeros(self.__model.vocabularyLength)

        #Inicializamos diccionario de frecuencias de palabras en el documento
        docTokensFreq = dict()
        tokens = word_tokenize(text)
        docTotalTokens = len(tokens)

        # Realizamos el conteo de frecuencias
        for token in tokens:
            if token not in docTokensFreq.keys():
                docTokensFreq[token] = 1
            else:
                docTokensFreq[token] += 1
        
        # Construimos el vector de cantidad de informacion
        for token in docTokensFreq.keys():
            vector[self.__model.getTokenId(token)] = -np.log2(docTokensFreq[token] / docTotalTokens)
        
        return vector

    def transform(self, corpusReader=None):
        """
        Tansforma un corpus en un conjunto de vectores de características léxicas.

        Args:
            CorpusReader (CorpusReader): Lector de documentos de texto. 
                Se utiliza en la etapa de pruebas. En caso de no espeficicar uno
                se hace uso del mismo lector de la etapa de entrenamiento ingresada
                en el constructor.                
        """

        gen = self.__corpusReader.getBatch() if corpusReader is None else corpusReader.getBatch()

        for batch in gen:
            vectors = []

            for t in batch[1]:
                print(f"Vectorizando documento {t['id']}", end='\r')
                v = dict()
                v['id'] = t['id']
                v['vector'] = self.__getVector(t['text'])
                vectors.append(v)
            self.__vectorWriter.addPreprocessedBatch((batch[0], vectors))
    
    def saveModel(self, outputPath):
        """Guarda el modelo de vectorización de documentos entrenado."""

        self.__model.save(outputPath)

class SyntacticVectorizer(Vectorizer):
    """
    Vectorizador sintácico de documentos de texto.

    Para esta versión de la aplicación se utiliza el mismo vectorizador semántico 
    (ver SemanticVectorizer), con la diferencia en que la entrada recibe
    cadenas de etiqutas POS. 
    """

    def fit(self):
        pass

    def transform(self, corpus):
        pass

    def saveModel(self, outputPath):
        pass

class SemanticVectorizer(Vectorizer):
    """
    Vectorizador semántico de documentos de texto.
    
    Attributes:
        vectorWriter (DocumentSink): Recolector de documentos vectorizados.
        corpusReader (CorpusReader): Lector de documentos de texto. Solamente se
            especifica en la etapa de entrenamiento del modelo de vectorización.
        vectorSize (int): Numero de dimensiones de los vectores de características
            de salida.
        minCount (int): Número mínimo de aparición de una palabra para ser considerada
            en el entrenamiento.
        epochs (int): Número de iteraciones en las que se entrenará
            el modelo de vectorización.
    """

    def __init__(self, vectorWriter, corpusReader=None, vectorSize=50, minCount=1, epochs=10):
        self.__model = Doc2Vec(vector_size=vectorSize, min_count=minCount, epochs=epochs)
        self.__vectorWriter = vectorWriter
        self.__corpusReader = corpusReader

    @property
    def model(self):
        """Modelo de vectorización."""
        return self.__model

    @model.setter
    def model(self, inputPath):
        """
        Estabece un modelo de vectorización pre-entrnado.
        
        Args:
            inputPath (str): Ruta del modelo a cargar.
        """

        self.__model = Doc2Vec.load(inputPath)
    
    def fit(self):
        """
        Entrena un modelo de vectorización léxico de documentos.
        """

        try:
            if self.__corpusReader is None:
                raise Exception('No se ha definido un corpus reader en el constructor para el entrenamiento.')
            else:
                self.__model.build_vocab(self.__corpusReader)
                self.__model.train(self.__corpusReader, total_examples=self.__model.corpus_count, epochs=self.__model.epochs)            
        except Exception as err:
            print(err)        

    def transform(self, corpusReader=None):
        """
        Tansforma un corpus en un conjunto de vectores de características semánticas.

        Args:
            CorpusReader (CorpusReader): Lector de documentos de texto. 
                Se utiliza en la etapa de pruebas. En caso de no espeficicar uno
                se hace uso del mismo lector de la etapa de entrenamiento ingresada
                en el constructor.
        """

        cr = self.__corpusReader if corpusReader is None else corpusReader

        for doc in cr:
            idDoc = doc[1][0]
            docVec = self.__model.dv[idDoc]
            batch = (idDoc,[{'id':idDoc, 'vector':docVec}])
            self.__vectorWriter.addPreprocessedBatch(batch)        

    def saveModel(self, outputPath):
        self.__model.save(outputPath)

class VectorizerAbstractFactory(metaclass=ABCMeta):
    """Fabrica abstracta de vectorizadores."""
    @abstractmethod
    def createLexicVectorizer(self):
        """Crea un vectorizador léxico."""
        pass

    @abstractmethod
    def createSyntacticVectorizer(self):
        """Crea un vectorizador sintáctico."""
        pass

    @abstractmethod
    def createSemanticVectorizer(self):
        """Crea un vectorizador semántico."""
        pass

class VectorizerFactory(VectorizerAbstractFactory):
    """Fabrica de vectorizadores"""

    def createLexicVectorizer(self, vectorWriter, corpusReader=None):
        """
        Crea un vectorizador léxico.

        Args:
            vectorWriter (DocumentSink): Recolector de documentos vectorizados.
            corpusReader (CorpusReader): Lector de documentos de texto. Solamente se
            utiliza en la etapa de entrenamiento del modelo de vectorización.
        
        Returns:
            LexicVectorizer
        """

        return LexicVectorizer(vectorWriter, corpusReader)

    def createSyntacticVectorizer(self, vectorWriter, corpusReader=None, vectorSize=50, minCount=1, epochs=10):
        """
        Crea un vectorizador sintáctico.

        Para esta versión de la aplicación se utiliza el mismo vectorizador semántico 
        (ver SemanticVectorizer), con la diferencia en que la entrada recibe
        cadenas de etiqutas POS.

        Args:
            vectorWriter (DocumentSink): Recolector de documentos vectorizados.
            corpusReader (CorpusReader): Lector de documentos de texto. Solamente se
                especifica en la etapa de entrenamiento del modelo de vectorización.
            vectorSize (int): Numero de dimensiones de los vectores de características
                de salida.
            minCount (int): Número mínimo de aparición de una palabra para ser considerada
                en el entrenamiento.
            epochs (int): Número de iteraciones en las que se entrenará
                el modelo de vectorización.

        Returns:
            SemanticVectorizer
        """

        return SemanticVectorizer(vectorWriter, corpusReader, vectorSize, minCount, epochs)

        
    def createSemanticVectorizer(self, vectorWriter, corpusReader=None, vectorSize=50, minCount=1, epochs=10):
        """
        Crea un vectorizador semántico.

        Args:
            vectorWriter (DocumentSink): Recolector de documentos vectorizados.
            corpusReader (CorpusReader): Lector de documentos de texto. Solamente se
                especifica en la etapa de entrenamiento del modelo de vectorización.
            vectorSize (int): Numero de dimensiones de los vectores de características
                de salida.
            minCount (int): Número mínimo de aparición de una palabra para ser considerada
                en el entrenamiento.
            epochs (int): Número de iteraciones en las que se entrenará
                el modelo de vectorización.

        Returns:
            SemanticVectorizer
        """
        return SemanticVectorizer(vectorWriter, corpusReader, vectorSize, minCount, epochs)

class LexicModel():
    """
    Modelo léxico de vectorización.
    
    Este modelo está basado en la candidad de información que aporta
    una palabra al documento.

    Attributes:
        vocabulary (dict): Diccionario de frecuencias de palabras en el corpus.
        corpusTotalTokens (int): Total de tokens en el corpus.
    """

    def __init__(self):
        """Inicializa el modelo léxico."""

        self.__vocabulary = dict()
        self.__corpusTotalTokens = 0

    @property
    def corpusTotalTokens(self):
        """Devuelve el total de tokens en el corpus."""
        return self.__corpusTotalTokens

    @property
    def vocabularyLength(self):
        """Devuelve la cantidad de palabras únicas en el corpus."""
        return len(self.__token2id.keys())

    def addToken(self, token):
        """
        Agrega un token al vocabulario y la conrabilización de palabras en el corpus.
        
        Args:
            token (str): Palabra a agregar.
        """

        if token not in self.__vocabulary.keys():
            self.__vocabulary[token] = 1
            self.__corpusTotalTokens += 1
        else:
            self.__vocabulary[token] += 1
            self.__corpusTotalTokens += 1

    def getTokenProbability(self, token):
        """
        Devuelve la probabilidad de aparición de una palabra en el corpus.

        Args:
            token (str): Palabra cuya probabiliad de aparición se desea saber.

        Returns:
            Probabilidad de la palabra.
        """

        if token in self.__vocabulary.keys():
            return self.__vocabulary[token] / self.__corpusTotalTokens
        else:
            return 0

    def getTokenFrequency(self, token):
        """
        Devuelve la frecuencia de la plabra en el corpus

        Args:
            token (str): Palabra cuya frecuencia de aparición se desea saber.
        
        Returns:
            Frecuencia de la palabra.
        """

        if token in self.__vocabulary.keys():
            return self.__vocabulary[token]
        else:
            return 0

    def getVocabulary(self):
        """
        Devuelve el vocabulario del corpus.
        """

        return self.__vocabulary.keys()

    def setToken2Id(self):
        """Establece un diccionario de palabras con un id único."""

        self.__token2id = dict()
        for idx, token in enumerate(self.__vocabulary.keys()):
            self.__token2id[token] = idx
    
    def getTokenId(self, token):
        """Devuelve el id de una palabra en el corpus."""

        try:
            if token not in self.__token2id.keys():
                raise Exception("No existe el token en el vocabulario")
            else:
                return self.__token2id[token]

        except Exception as err:
            print(err)

    def save(self, outputPath):
        """Guarda el modelo léxico."""

        with open(outputPath, 'w', encoding='utf-8') as f:
            dumpDict = {'corpusTotalTokens': self.__corpusTotalTokens,
                'tokenFreq': self.__vocabulary,
                'token2id': self.__token2id}
            f.write(json.dumps(dumpDict))

    def load(self, inputPath):
        """Carga un modelo léxico preestablecido."""
        
        with open(inputPath, encoding='utf-8') as f:
            dumpDict = json.loads(f.read())
            self.__corpusTotalTokens = dumpDict['corpusTotalTokens']
            self.__vocabulary = dumpDict['tokenFreq']
            self.__token2id = dumpDict['token2id']