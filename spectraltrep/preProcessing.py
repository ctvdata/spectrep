from abc import ABCMeta
from abc import abstractmethod
from threading import Thread
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize  
from nltk.tokenize import RegexpTokenizer
import re
from nltk.stem import WordNetLemmatizer
wordnet.ensure_loaded()
from pathlib import Path
from spectraltrep.utils import DocumentSink, Sink
from spectraltrep.utils import LockedIterator, CorpusReader

class Preprocessor(metaclass=ABCMeta):
    """Interfaz de preprocesamiento de texto"""
    @abstractmethod
    def preProcess(self, text):
        """
        Aplica una serie de tareas de preprocesamiento al texto de entrada.

        Args:
            text (str): Texto a preprocesar.
        """
        pass

class LexicPreprocessor(Preprocessor, Thread):
    """
    Hilo de preprocesamiento léxico de texto.
    
    Attributes:
        dispatcher (LockedIterator): Objeto de sincronización de generadores
        para hilos de preprocesamiento.
        
        sink (DocumentSink): Objeto de recolección de documentos preprocesados.
    """

    def __init__(self, dispatcher, sink):
        """Inicializa el hilo de preprocesamiento léxico."""

        Thread.__init__(self)
        try:
            if isinstance(dispatcher, LockedIterator):
                self.__dispatcher = dispatcher
            else:
                raise Exception("Non-valid instance of dispatcher.")

            if isinstance(sink, Sink):
                self.__sink = sink
            else:
                raise Exception("Non-valid instance of Sink.")
        
        except Exception as err:
            print(err)
       
        # self.__DELETE_NEW_LINE = re.compile('\\n') # Reemplazo de saltos de linea
        # self.__DELETE_MIDSCORE = re.compile('-') # Eliminacion de guion medio
        # self.__DELETE_PARENTHESES = re.compile('\(|\)') # Eliminación de paréntesis
        # self.__DELETE_BRACKETS = re.compile('\[|\]') # Eliminacion de corchetes
        # self.__REPLACE_DOUBLE_SPACE = re.compile('\s+') # Remplazo de espacios dobles
        # self.__DELETE_QM = re.compile('"|’|\'') # Eliminar comillas
        # self.__DELETE_PUNCTUATION = re.compile('[^\w\s]') # Eliminacion de signos de puntuacion
        # self.__REPLACE_DIGITS = re.compile('\d') # Reemplazo de digitos
        # self.__stop_words = set(stopwords.words('english'))
        # self.__wordnet_lemmatizer = WordNetLemmatizer()

        # Funciones agregadas para RegexpTokenizer
        self.__exp1 = re.compile('><')
        self.__exp2 = re.compile('<')
        self.__exp3 = re.compile('>')
    
    def preProcess(self, text):
        """
        Aplica una serie de tareas de preprocesamiento al texto de entrada.

        Args:
            text (str): Texto a preprocesar.

        Returns:
            Texto preprocesado.
        """

        text = text.lower()
        # text = self.__DELETE_NEW_LINE.sub("", text)
        # text = self.__DELETE_MIDSCORE.sub(" ", text)
        # text = self.__DELETE_PARENTHESES.sub("", text)    
        # text = self.__DELETE_BRACKETS.sub("", text)    
        # text = self.__REPLACE_DOUBLE_SPACE.sub(" ", text)   
        # text = self.__DELETE_QM.sub("", text)
        # text = self.__DELETE_PUNCTUATION.sub('', text)
        # text = self.__REPLACE_DIGITS.sub('<NUM>', text)
                
        # word_tokens = word_tokenize(text)
        # text = [w for w in word_tokens if not w in self.__stop_words]
        # text = [self.__wordnet_lemmatizer.lemmatize(w) for w in text]

        # Agregado para usar RegexpTokenizer
        text = self.__exp1.sub("> <", text)
        text = self.__exp2.sub(" <", text)
        text = self.__exp3.sub("> ", text)
        tokenizer = RegexpTokenizer('\w+|<\w*?>|\S+')
        text = tokenizer.tokenize(text)

        text = ' '.join(text)    
        
        return text

    def run(self):
        """Inicia el hilo de preprocesamiento."""

        while(True):
            batch = next(self.__dispatcher)
            if(batch != '<EOC>'):
                
                documents = []

                for t in batch[1]:
                    t['text'] = self.preProcess(t['text'])
                    documents.append(t)
                self.__sink.addPreprocessedBatch((batch[0], documents))
            else:
                break

class SyntacticPreprocessor(Preprocessor, Thread):
    """
    Hilo de preprocesamiento sintáctico de texto
    
    Attributes:
        dispatcher (LockedIterator): Objeto de sincronización de generadores
        para hilos de preprocesamiento.

        sink (DocumentSink): Objeto de recolección de documentos preprocesados.
    """

    def __init__(self, dispatcher, sink):
        """
        Inicializa el hilo de preprocesamiento léxico

        Args:
            dispatcher (LockedIterator): Objeto de sincronización de generadores
            para hilos de preprocesamiento.

            sink (DocumentSink): Objeto de recolección de documentos preprocesados.        
        """

        Thread.__init__(self)
        try:
            if isinstance(dispatcher, LockedIterator):
                self.__dispatcher = dispatcher
            else:
                raise Exception("Non-valid instance of dispatcher.")

            if isinstance(sink, Sink):
                self.__sink = sink
            else:
                raise Exception("Non-valid instance of Sink.")
        
        except Exception as err:
            print(err)

        # self.__DELETE_NEW_LINE = re.compile('\\n') # Reemplazo de saltos de linea
        # self.__DELETE_MIDSCORE = re.compile('-') # Eliminacion de guion medio
        # self.__DELETE_PARENTHESES = re.compile('\(|\)') # Eliminación de paréntesis
        # self.__DELETE_BRACKETS = re.compile('\[|\]') # Eliminacion de corchetes
        # self.__REPLACE_DOUBLE_SPACE = re.compile('\s+') # Remplazo de espacios dobles
        # self.__DELETE_QM = re.compile('"|’|\'') # Eliminar comillas
        # self.__DELETE_PUNCTUATION = re.compile('[^\w\s]') # Eliminacion de signos de puntuacion
        # self.__REPLACE_DIGITS = re.compile('\d') # Reemplazo de digitos

        # Funciones agregadas para RegexpTokenizer
        self.__exp1 = re.compile('><')
        self.__exp2 = re.compile('<')
        self.__exp3 = re.compile('>')
    
    def preProcess(self, text):
        """
        Aplica una serie de tareas de preprocesamiento al texto de entrada.

        Args:
            text (str): Texto a preprocesar.

        Returns:
            Texto preprocesado.
        """

        text = text.lower()
        # text = self.__DELETE_NEW_LINE.sub("", text)
        # text = self.__DELETE_MIDSCORE.sub(" ", text)
        # text = self.__DELETE_PARENTHESES.sub("", text)    
        # text = self.__DELETE_BRACKETS.sub("", text)    
        # text = self.__REPLACE_DOUBLE_SPACE.sub(" ", text)   
        # text = self.__DELETE_QM.sub("", text)
        # text = self.__DELETE_PUNCTUATION.sub('', text)
        # text = self.__REPLACE_DIGITS.sub('<NUM>', text)
        # text = word_tokenize(text)

        # Agregado para usar RegexpTokenizer
        text = self.__exp1.sub("> <", text)
        text = self.__exp2.sub(" <", text)
        text = self.__exp3.sub("> ", text)
        tokenizer = RegexpTokenizer('\w+|<\w*?>|\S+')
        text = tokenizer.tokenize(text)

        text = nltk.pos_tag(text)
        text = ' '.join([t[1] for t in text])    
        
        return text

    def run(self):
        """Inicia el hilo de preprocesamiento."""

        while(True):
            batch = next(self.__dispatcher)
            if(batch != '<EOC>'):                
                documents = []

                for t in batch[1]:
                    t['text'] = self.preProcess(t['text'])
                    documents.append(t)
                self.__sink.addPreprocessedBatch((batch[0], documents))
            else:
                break

class SemanticPreprocessor(Preprocessor, Thread):
    """
    Hilo de preprocesamiento semántico de texto
    
    Attributes:
        dispatcher (LockedIterator): Objeto de sincronización de generadores
        para hilos de preprocesamiento.

        sink (DocumentSink): Objeto de recolección de documentos preprocesados.
    """

    def __init__(self, dispatcher, sink):
        Thread.__init__(self)
        try:
            if isinstance(dispatcher, LockedIterator):
                self.__dispatcher = dispatcher
            else:
                raise Exception("Non-valid instance of dispatcher.")

            if isinstance(sink, Sink):
                self.__sink = sink
            else:
                raise Exception("Non-valid instance of Sink.")
        
        except Exception as err:
            print(err)
            
        # self.__DELETE_NEW_LINE = re.compile('\\n') # Reemplazo de saltos de linea
        # self.__DELETE_MIDSCORE = re.compile('-') # Eliminacion de guion medio
        # self.__DELETE_PARENTHESES = re.compile('\(|\)') # Eliminación de paréntesis
        # self.__DELETE_BRACKETS = re.compile('\[|\]') # Eliminacion de corchetes
        # self.__REPLACE_DOUBLE_SPACE = re.compile('\s+') # Remplazo de espacios dobles
        # self.__DELETE_QM = re.compile('"|’|\'') # Eliminar comillas
        # self.__DELETE_PUNCTUATION = re.compile('[^\w\s]') # Eliminacion de signos de puntuacion
        # self.__REPLACE_DIGITS = re.compile('\d') # Reemplazo de digitos
        # self.__stop_words = set(stopwords.words('english'))

        # Funciones agregadas para RegexpTokenizer
        self.__exp1 = re.compile('><')
        self.__exp2 = re.compile('<')
        self.__exp3 = re.compile('>')
    
    def preProcess(self, text):
        """
        Aplica una serie de tareas de preprocesamiento al texto de entrada.

        Args:
            text (str): Texto a preprocesar.
        Returns:
            Texto preprocesado.
        """

        text = text.lower()
        # text = self.__DELETE_NEW_LINE.sub("", text)
        # text = self.__DELETE_MIDSCORE.sub(" ", text)
        # text = self.__DELETE_PARENTHESES.sub("", text)    
        # text = self.__DELETE_BRACKETS.sub("", text)    
        # text = self.__REPLACE_DOUBLE_SPACE.sub(" ", text)   
        # text = self.__DELETE_QM.sub("", text)
        # text = self.__DELETE_PUNCTUATION.sub('', text)
        # text = self.__REPLACE_DIGITS.sub('<NUM>', text)
        
        # word_tokens = word_tokenize(text)
        # text = [w for w in word_tokens if not w in self.__stop_words]
        text = self.__exp1.sub("> <", text)
        text = self.__exp2.sub(" <", text)
        text = self.__exp3.sub("> ", text)
        tokenizer = RegexpTokenizer('\w+|<\w*?>|\S+')
        text = tokenizer.tokenize(text)
        
        text = ' '.join(text)
        
        return text

    def run(self):
        """Inicia el hilo de preprocesamiento."""

        while(True):
            batch = next(self.__dispatcher)
            if(batch != '<EOC>'):
                
                documents = []

                for t in batch[1]:
                    t['text'] = self.preProcess(t['text'])
                    documents.append(t)
                self.__sink.addPreprocessedBatch((batch[0], documents))
            else:
                break

class PreprocessorAbstractFactory(metaclass=ABCMeta):
    """
    Fabrica abstracta de hilos de preprocesamiento.
    """

    @abstractmethod
    def createLexicPreprocessor(self, dispatcher, sink):
        """
        Crea un hilo de preprocesamiento léxico.

        Args:
            dispatcher (LockedIterator): Objeto de sincronización de generadores
            para hilos de preprocesamiento.

            sink (DocumentSink): Objeto de recolección de documentos preprocesados.
        """
        pass

    @abstractmethod
    def createSyntacticPreprocessor(self, dispatcher, sink):
        """
        Crea un hilo de preprocesamiento sintáctico.

        Args:
            dispatcher (LockedIterator): Objeto de sincronización de generadores
            para hilos de preprocesamiento.

            sink (DocumentSink): Objeto de recolección de documentos preprocesados.
        """
        pass
    
    @abstractmethod
    def createSemanticPreprocessor(self, dispatcher, sink):
        """
        Crea un hilo de preprocesamiento semántico.

        Args:
            dispatcher (LockedIterator): Objeto de sincronización de generadores
            para hilos de preprocesamiento.

            sink (DocumentSink): Objeto de recolección de documentos preprocesados.
        """
        pass

class PreprocessorFactory(PreprocessorAbstractFactory):
    """
    Fabrica de hilos de preprocesamiento.
    """

    def createLexicPreprocessor(self, dispatcher, sink):
        """
        Crea un hilo de preprocesamiento léxico.

        Args:
            dispatcher (LockedIterator): Objeto de sincronización de generadores
            para hilos de preprocesamiento.

            sink (DocumentSink): Objeto de recolección de documentos preprocesados.

        Returns:
            Hilo de preprocesamiento léxico.
        """
        return LexicPreprocessor(dispatcher, sink)
    
    def createSyntacticPreprocessor(self, dispatcher, sink):
        """
        Crea un hilo de preprocesamiento sintáctico.

        Args:
            dispatcher (LockedIterator): Objeto de sincronización de generadores
            para hilos de preprocesamiento.

            sink (DocumentSink): Objeto de recolección de documentos preprocesados.

        Returns:
            Hilo de preprocesamiento sintáctico.
        """
        return SyntacticPreprocessor(dispatcher, sink)

    def createSemanticPreprocessor(self, dispatcher, sink):
        """
        Crea un hilo de preprocesamiento semántico.

        Args:
            dispatcher (LockedIterator): Objeto de sincronización de generadores
            para hilos de preprocesamiento.

            sink (DocumentSink): Objeto de recolección de documentos preprocesados.

        Returns:
            Hilo de preprocesamiento semántico.
        """
        return SemanticPreprocessor(dispatcher, sink)

class PreProcessingFacade():
    """
    Fachada de preprocesamiento

    Abstrae la lógica de crear un pipeline de preprocesamiento
    léxico, sintáctico y semántico.
    """
    
    def preProcess(self, input, output, preProcessingType=["lex", "syn", "sem"], numThreads=1, batchSize=3000, sortedOutput=True):
        """
        Realiza el preprocesamiento de un corpus.

        Args:
            input (str): Ruta del archivo de entrada.

            output (str): Ruta del archivo de salida.

            preProcessingType (list): Lista de tipos de

            preprocesamiento a realizar ["lex", "syn", "sem"].

            numThreads (int): Número de hilos de preprocesamiento.

            batchSize (int): Tamaño del lote de documentos por hilo de preprocesamiento.

            sortedOutput (bool): Determina si los documentos del corpus preprocesado
            tendrán el mismo orden de entrada.
        """
        ppf = PreprocessorFactory()
        
        try:
            for tp in preProcessingType:
                cr = CorpusReader(input, batchSize).getBatch()
                lockedCr = LockedIterator(cr)
                p = Path(output)
                fileName = '{}_{}{}'.format(p.stem, tp, p.suffix)
                ds = DocumentSink(Path(p.parent, fileName), sortedOutput)

                # Inicializamos hilos de preprocesamiento
                PreprocessingThreads = []
                for _ in range(numThreads):
                    if tp=="lex":
                        pp = ppf.createLexicPreprocessor(lockedCr, ds)
                    elif tp=="syn":
                        pp = ppf.createSyntacticPreprocessor(lockedCr, ds)
                    elif tp=="sem":
                        pp = ppf.createSemanticPreprocessor(lockedCr, ds)
                    else:
                        raise Exception("Tipo de preprocesamiento no valido.")
                    
                    pp.start()
                    PreprocessingThreads.append(pp)

                # Esperamos a que terminen los hilos de preprocesamiento
                for t in PreprocessingThreads:
                    t.join()

                # Guardamos el corpus preprocesado en caso de que sea un corpus ordenado
                if sortedOutput:
                    ds.saveCorpus()

        except Exception as err:
            print(err)