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
from nltk.tokenize import word_tokenize  
import re
from nltk.stem import WordNetLemmatizer
wordnet.ensure_loaded()
from pathlib import Path
from spectraltrep.utils import DocumentSink, Sink
from spectraltrep.utils import LockedIterator, CorpusReader

class Preprocessor(metaclass=ABCMeta):
    """Text pre-processing interface"""
    @abstractmethod
    def preProcess(self, text):
        """
        Applies a series of pre-processing tasks to the input text.

        Args:
            text (str): Text to pre-process.
        """
        pass

class LexicPreprocessor(Preprocessor, Thread):
    """
    Text lexical pre-processing thread.
    
    Attributes:
        dispatcher (LockedIterator): Generator synchronization object for pre-processing threads.
        
        sink (DocumentSink): Pre-processed document sink.
    """

    def __init__(self, dispatcher, sink):
        """Initializes the lexical pre-processing thread."""

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
       
        self.__DELETE_NEW_LINE = re.compile('\\n') # Line break replacement
        self.__DELETE_MIDSCORE = re.compile('-') # Mid dash removal
        self.__DELETE_PARENTHESES = re.compile('\(|\)') # Removing parentheses
        self.__DELETE_BRACKETS = re.compile('\[|\]') # Removing brackets
        self.__REPLACE_DOUBLE_SPACE = re.compile('\s+') # Replacing double spaces
        self.__DELETE_QM = re.compile('"|’|\'') # Remove quotes
        self.__DELETE_PUNCTUATION = re.compile('[^\w\s]') # Elimination of punctuation marks
        self.__REPLACE_DIGITS = re.compile('\d') # Digit Replacement
        self.__stop_words = set(stopwords.words('english'))
        self.__wordnet_lemmatizer = WordNetLemmatizer()
    
    def preProcess(self, text):
        """
        Applies a series of pre-processing tasks to the input text.

        Args:
            text (str): Text to pre-process.

        Returns:
            Pre-processed text.
        """

        text = text.lower()
        text = self.__DELETE_NEW_LINE.sub("", text)
        text = self.__DELETE_MIDSCORE.sub(" ", text)
        text = self.__DELETE_PARENTHESES.sub("", text)    
        text = self.__DELETE_BRACKETS.sub("", text)    
        text = self.__REPLACE_DOUBLE_SPACE.sub(" ", text)   
        text = self.__DELETE_QM.sub("", text)
        text = self.__DELETE_PUNCTUATION.sub('', text)
        text = self.__REPLACE_DIGITS.sub('<NUM>', text)
        
        word_tokens = word_tokenize(text)
        text = [w for w in word_tokens if not w in self.__stop_words]
        text = [self.__wordnet_lemmatizer.lemmatize(w) for w in text]
        
        text = ' '.join(text)    
        
        return text

    def run(self):
        """Starts the pre-processing thread."""

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
    Text syntactical pre-processing thread.
    
    Attributes:
        dispatcher (LockedIterator): Generator synchronization object for pre-processing threads.

        sink (DocumentSink): Pre-processed document sink.
    """

    def __init__(self, dispatcher, sink):
        """Initializes the syntactical pre-processing thread."""

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

        self.__DELETE_NEW_LINE = re.compile('\\n') # Line break replacement
        self.__DELETE_MIDSCORE = re.compile('-') # Mid-hyphen removal
        self.__DELETE_PARENTHESES = re.compile('\(|\)') # Removing parentheses
        self.__DELETE_BRACKETS = re.compile('\[|\]') # Removing brackets
        self.__REPLACE_DOUBLE_SPACE = re.compile('\s+') # Replacing double spaces
        self.__DELETE_QM = re.compile('"|’|\'') # Remove quotes
        self.__DELETE_PUNCTUATION = re.compile('[^\w\s]') # Elimination of punctuation marks
        self.__REPLACE_DIGITS = re.compile('\d') # Digit Replacement
    
    def preProcess(self, text):
        """
        Applies a series of pre-processing tasks to the input text.

        Args:
            text (str): Text to pre-process.

        Returns:
            Pre-processed text
        """

        text = text.lower()
        text = self.__DELETE_NEW_LINE.sub("", text)
        text = self.__DELETE_MIDSCORE.sub(" ", text)
        text = self.__DELETE_PARENTHESES.sub("", text)    
        text = self.__DELETE_BRACKETS.sub("", text)    
        text = self.__REPLACE_DOUBLE_SPACE.sub(" ", text)   
        text = self.__DELETE_QM.sub("", text)
        text = self.__DELETE_PUNCTUATION.sub('', text)
        text = self.__REPLACE_DIGITS.sub('<NUM>', text)
        text = word_tokenize(text)
        text = nltk.pos_tag(text)
        text = ' '.join([t[1] for t in text])    
        
        return text

    def run(self):
        """Starts the pre-processing thread."""

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
    Semantic text pre-processing thread.
    
    Attributes:
        dispatcher (LockedIterator): Generator synchronization object for pre-processing threads.

        sink (DocumentSink): Pre-processed document sink.
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
            
        self.__DELETE_NEW_LINE = re.compile('\\n') # Line break replacement
        self.__DELETE_MIDSCORE = re.compile('-') # Mid-hyphen removal
        self.__DELETE_PARENTHESES = re.compile('\(|\)') # Removing parentheses
        self.__DELETE_BRACKETS = re.compile('\[|\]') # Removing brackets
        self.__REPLACE_DOUBLE_SPACE = re.compile('\s+') # Replacing double spaces
        self.__DELETE_QM = re.compile('"|’|\'') # Remove quotes
        self.__DELETE_PUNCTUATION = re.compile('[^\w\s]') # Elimination of punctuation marks
        self.__REPLACE_DIGITS = re.compile('\d') # Digit Replacement
        self.__stop_words = set(stopwords.words('english'))
    
    def preProcess(self, text):
        """
        Applies a series of pre-processing tasks to the input text.

        Args:
            text (str): Text to pre-process.
        Returns:
            Pre-processed text.
        """

        text = text.lower()
        text = self.__DELETE_NEW_LINE.sub("", text)
        text = self.__DELETE_MIDSCORE.sub(" ", text)
        text = self.__DELETE_PARENTHESES.sub("", text)    
        text = self.__DELETE_BRACKETS.sub("", text)    
        text = self.__REPLACE_DOUBLE_SPACE.sub(" ", text)   
        text = self.__DELETE_QM.sub("", text)
        text = self.__DELETE_PUNCTUATION.sub('', text)
        text = self.__REPLACE_DIGITS.sub('<NUM>', text)
        
        word_tokens = word_tokenize(text)
        text = [w for w in word_tokens if not w in self.__stop_words]
        text = ' '.join(text)
        
        return text

    def run(self):
        """Starts the pre-processing thread."""

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
    Pre-processing threads abstract factory.
    """

    @abstractmethod
    def createLexicPreprocessor(self, dispatcher, sink):
        """
        Creates a lexical pre-processing thread.

        Args:
            dispatcher (LockedIterator): Generator synchronization object for pre-processing threads.

            sink (DocumentSink): Pre-processed document sink.
        """
        pass

    @abstractmethod
    def createSyntacticPreprocessor(self, dispatcher, sink):
        """
        Creates a syntactical pre-processing thread.

        Args:
            dispatcher (LockedIterator): Generator synchronization object for pre-processing threads.

            sink (DocumentSink): Pre-processed document sink.
        """
        pass
    
    @abstractmethod
    def createSemanticPreprocessor(self, dispatcher, sink):
        """
        Creates a semantic pre-processing thread.

        Args:
            dispatcher (LockedIterator): Generator synchronization object for pre-processing threads.

            sink (DocumentSink): Pre-processed document sink.
        """
        pass

class PreprocessorFactory(PreprocessorAbstractFactory):
    """
    Pre-processing threads factory.
    """

    def createLexicPreprocessor(self, dispatcher, sink):
        """
        Creates a lexical pre-processing thread.

        Args:
            dispatcher (LockedIterator): Generator synchronization object for pre-processing threads.

            sink (DocumentSink): Pre-processed document sink.

        Returns:
            Lexical pre-processing thread.
        """
        return LexicPreprocessor(dispatcher, sink)
    
    def createSyntacticPreprocessor(self, dispatcher, sink):
        """
        Creates a syntactical pre-processing thread.

        Args:
            dispatcher (LockedIterator): Generator synchronization object for pre-processing threads.

            sink (DocumentSink): Pre-processed document sink.

        Returns:
            Syntactical pre-processing thread.
        """
        return SyntacticPreprocessor(dispatcher, sink)

    def createSemanticPreprocessor(self, dispatcher, sink):
        """
        Creates a semantic pre-processing thread.

        Args:
            dispatcher (LockedIterator): Generator synchronization object for pre-processing threads.

            sink (DocumentSink): Pre-processed document sink.

        Returns:
            Semantic pre-processing thread.
        """
        return SemanticPreprocessor(dispatcher, sink)

class PreProcessingFacade():
    """
    Pre-processing facade.

    Abstracts the logic of creating a lexical, syntactical and semantic pre-processing pipeline. 
    """
    
    def preProcess(self, input, output, preProcessingType=["lex", "syn", "sem"], numThreads=1, batchSize=3000, sortedOutput=True):
        """
        Pre-processing a corpus. 

        Args:
            input (str): File input path.

            output (str): File output path.

            preProcessingType (list): List of types of preprocessing to be performed ["lex", "syn", "sem"].

            numThreads (int): Number of pre-processing threads.

            batchSize (int): Number of documents per batch.

            sortedOutput (bool): Determines if the pre-processed corpus will have the same order at its output.
        """
        ppf = PreprocessorFactory()
        
        try:
            for tp in preProcessingType:
                cr = CorpusReader(input, batchSize).getBatch()
                lockedCr = LockedIterator(cr)
                p = Path(output)
                fileName = '{}_{}{}'.format(p.stem, tp, p.suffix)
                ds = DocumentSink(Path(p.parent, fileName), sortedOutput)

                # e initialize preprocessing threads
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

                # Wait for the preprocessing threads to finish
                for t in PreprocessingThreads:
                    t.join()

                # We save the preprocessed corpus in case it is an ordered corpus
                if sortedOutput:
                    ds.saveCorpus()

        except Exception as err:
            print(err)