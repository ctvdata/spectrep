from abc import ABCMeta, abstractmethod
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json
from nltk.tokenize import word_tokenize
import json
import numpy as np

class Doc2VecCorpusReader():
    """
    Corpus reader generator

    Used by the GensSim Doc2Vec class

    Args:
            inputPath (str): File input path.

    Yields:
        TaggedDocument
    """

    def __init__(self, inputPath):
        """
        Initializes the document generator.

        Args:
            inputPath (str): File input path.
        """
        
        self.__inputPath = inputPath

    def __iter__(self):        
        with open(self.__inputPath) as f:
            for line in f:
                line = json.loads(line)
                tokens = word_tokenize(line['text'])
                yield TaggedDocument(tokens, [int(line['id'])])

class Vectorizer(metaclass=ABCMeta):
    """Document vectorizer interface"""

    @abstractmethod
    def fit(self):
        """
        Fits a document vectorization model.
        """
        pass

    @abstractmethod
    def transform(self, corpus):
        """
        Transforms a corpus into a set of feature vectors.
        """
        pass

    @abstractmethod
    def saveModel(self, outputPath):
        """Saves the trained document vectorizer model."""
        pass

class LexicVectorizer(Vectorizer):
    """
    Lexic document vectorizer.
    
    Args:
        vectorWriter (DocumentSink): Vectorized documents sink.
        
        corpusReader (CorpusReader): Document reader. It is only used in the
        training stage of the vectorization model.
    """

    def __init__(self, vectorWriter, corpusReader=None):
        """Initialize the lexical vectorizer."""

        self.__model = LexicModel()
        self.__vectorWriter = vectorWriter
        self.__corpusReader = corpusReader

    @property
    def model(self):
        """Vectorization model."""
        return self.__model

    @model.setter
    def model(self, inputPath):
        """
        Establishes a pre-trained vectorization model.
        
        Args:
            inputPath (str): Path of the model to load.
        """
        model = LexicModel()
        model.load(inputPath)
        self.__model = model

    def fit(self):
        """
        Trains a lexical vectorization model.
        """

        try:
            if self.__corpusReader is None:
                raise Exception('A reader corpus has not been defined in the constructor.')
            else:
                gen = self.__corpusReader.getBatch()
                for batch in gen:
                    for doc in batch[1]:
                        print(f"Training with doc {doc['id']}", end='\r')
                        doc = word_tokenize(doc['text'])
                        for token in doc:
                            self.__model.addToken(token)
                self.__model.setToken2Id()

        except Exception as err:
            print(err)

    def __getVector(self, text):
        # We create the vector of zeros
        vector = np.zeros(self.__model.vocabularyLength)

        # We initialize dictionary of word frequencies in the document
        docTokensFreq = dict()
        tokens = word_tokenize(text)
        docTotalTokens = len(tokens)

        # We perform the frequency count
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
        Transforms a corpus into a set of lexical feature vectors.

        Args:
            CorpusReader (CorpusReader): Text document reader. If one is not specified, 
            the same reader of the training stage set in the 
            constructor is used.                
        """

        gen = self.__corpusReader.getBatch() if corpusReader is None else corpusReader.getBatch()

        for batch in gen:
            vectors = []

            for t in batch[1]:
                print(f"Vectorizing document {t['id']}", end='\r')
                v = dict()
                v['id'] = t['id']
                v['vector'] = self.__getVector(t['text'])
                vectors.append(v)
            self.__vectorWriter.addPreprocessedBatch((batch[0], vectors))
    
    def saveModel(self, outputPath):
        """
        Saves the trained document vectorization model.

        Args:
            outputPath (str): File output path.
        """

        self.__model.save(outputPath)

class SyntacticVectorizer(Vectorizer):
    """
    Syntactic document vectorizer.

    For this version of the application the same semantic vectorizer is used 
    (see SemanticVectorizer), with the difference that the input 
    receives strings of POS tags. 
    """

    def fit(self):
        pass

    def transform(self, corpus):
        pass

    def saveModel(self, outputPath):
        pass

class SemanticVectorizer(Vectorizer):
    """
    Semantic document vectorizer.
    
    Args:
        vectorWriter (DocumentSink): Vectorized documents sink.

        corpusReader (CorpusReader): Text document reader. It is only specified
        during the vectorization model training stage.

        vectorSize (int): Number of dimensions of the output feature vectors.

        minCount (int): The minimum number of word appearances
        to be considered in the training.

        epochs (int): Number of iterations in which the vectorization model
        will be trained.
    """

    def __init__(self, vectorWriter, corpusReader=None, vectorSize=50, minCount=1, epochs=10):
        """Initializes the semantic vectorizer."""
        self.__model = Doc2Vec(vector_size=vectorSize, min_count=minCount, epochs=epochs)
        self.__vectorWriter = vectorWriter
        self.__corpusReader = corpusReader

    @property
    def model(self):
        """Vectorization model."""
        return self.__model

    @model.setter
    def model(self, inputPath):
        """
        Sets a pre-trainer vectorization model.
        
        Args:
            inputPath (str): Model input path.
        """

        self.__model = Doc2Vec.load(inputPath)
    
    def fit(self):
        """
        Train a document semantic vectorization model.
        """

        try:
            if self.__corpusReader is None:
                raise Exception('A reader corpus has not been defined in the constructor.')
            else:
                self.__model.build_vocab(self.__corpusReader)
                self.__model.train(self.__corpusReader, total_examples=self.__model.corpus_count, epochs=self.__model.epochs)            
        except Exception as err:
            print(err)        

    def transform(self, corpusReader=None):
        """
        Transforms a corpus into a set of semantic feature vectors.

        Args:
            CorpusReader (CorpusReader): Text document reader. If one is not specified, 
            the same reader of the training stage set in the 
            constructor is used.
        """

        cr = self.__corpusReader if corpusReader is None else corpusReader

        for doc in cr:
            idDoc = doc[1][0]
            docVec = self.__model.dv[idDoc]
            batch = (idDoc,[{'id':idDoc, 'vector':docVec}])
            self.__vectorWriter.addPreprocessedBatch(batch)        

    def saveModel(self, outputPath):
        """
        Saves the trained document vectorization model.

        Args:
            outputPath (str): File output path.
        """

        self.__model.save(outputPath)

class VectorizerAbstractFactory(metaclass=ABCMeta):
    """Vectorizer abstract factory."""
    @abstractmethod
    def createLexicVectorizer(self):
        """Creates a lexic vectorizer."""
        pass

    @abstractmethod
    def createSyntacticVectorizer(self):
        """Creates a syntactic vectorizer."""
        pass

    @abstractmethod
    def createSemanticVectorizer(self):
        """Creates a semantic vectorizer."""
        pass

class VectorizerFactory(VectorizerAbstractFactory):
    """Vectorizer factory."""

    def createLexicVectorizer(self, vectorWriter, corpusReader=None):
        """
        Creates a lexic vectorizer.

        Args:
            vectorWriter (DocumentSink): Vectorized document sink.

            corpusReader (CorpusReader): Text document reader. It is only used
            in the training stage of the vectorization model.
        
        Returns:
            LexicVectorizer
        """

        return LexicVectorizer(vectorWriter, corpusReader)

    def createSyntacticVectorizer(self, vectorWriter, corpusReader=None, vectorSize=50, minCount=1, epochs=10):
        """
        Creates a lexic vectorizer.

        For this version of the application the same semantic vectorizer is used 
        (see SemanticVectorizer), with the difference that the input 
        receives strings of POS tags. 

        Args:
            vectorWriter (DocumentSink): Vectorized documents sink.

            corpusReader (CorpusReader): Text document reader. It is only specified
            during the vectorization model training stage.

            vectorSize (int): Number of dimensions of the output feature vectors.

            minCount (int): The minimum number of word appearances
            to be considered in the training.

            epochs (int): Number of iterations in which the vectorization model
            will be trained.

        Returns:
            SemanticVectorizer
        """

        return SemanticVectorizer(vectorWriter, corpusReader, vectorSize, minCount, epochs)

        
    def createSemanticVectorizer(self, vectorWriter, corpusReader=None, vectorSize=50, minCount=1, epochs=10):
        """
        Creates a semantic vectorizer.

        Args:
            vectorWriter (DocumentSink): Vectorized documents sink.

            corpusReader (CorpusReader): Text document reader. It is only specified
            during the vectorization model training stage.

            vectorSize (int): Number of dimensions of the output feature vectors.

            minCount (int): The minimum number of word appearances
            to be considered in the training.

            epochs (int): Number of iterations in which the vectorization model
            will be trained.

        Returns:
            SemanticVectorizer
        """
        return SemanticVectorizer(vectorWriter, corpusReader, vectorSize, minCount, epochs)

class LexicModel():
    """
    Vectorization lexical model.
    
    This model is based on the amount of information that
    a word provides to a document.

    Args:
        vocabulary (dict): Corpus' dictionary of word frequencies.
        corpusTotalTokens (int): Total de tokens en el corpus.
    """

    def __init__(self):
        """Initializes the lexical model."""

        self.__vocabulary = dict()
        self.__corpusTotalTokens = 0

    @property
    def corpusTotalTokens(self):
        """Returns the corpus' total of tokens"""
        return self.__corpusTotalTokens

    @property
    def vocabularyLength(self):
        """Returns the vocabulary length."""
        return len(self.__token2id.keys())

    def addToken(self, token):
        """
        Adds a token to the corpus' vocabulary and its frequency.

        Args:
            token (str): Token to add.
        """

        if token not in self.__vocabulary.keys():
            self.__vocabulary[token] = 1
            self.__corpusTotalTokens += 1
        else:
            self.__vocabulary[token] += 1
            self.__corpusTotalTokens += 1

    def getTokenProbability(self, token):
        """
        Returns the probability of a word occurrence in the corpus.

        Args:
            token (str): Word whose probability of occurrence its wanted to know.

        Returns:
            Word probability.
        """

        if token in self.__vocabulary.keys():
            return self.__vocabulary[token] / self.__corpusTotalTokens
        else:
            return 0

    def getTokenFrequency(self, token):
        """
        Returns a word frequency in the corpus.

        Args:
            token (str): Word whose frequency its wanted to know.
        
        Returns:
            Word frequency.
        """

        if token in self.__vocabulary.keys():
            return self.__vocabulary[token]
        else:
            return 0

    def getVocabulary(self):
        """
        Returns the corpus' vocabulary
        """

        return self.__vocabulary.keys()

    def setToken2Id(self):
        """Sets a token dictionary with unique ids."""

        self.__token2id = dict()
        for idx, token in enumerate(self.__vocabulary.keys()):
            self.__token2id[token] = idx
    
    def getTokenId(self, token):
        """Returns the id of a token in the corpus."""

        try:
            if token not in self.__token2id.keys():
                raise Exception("The token does not exist in the vocabulary.")
            else:
                return self.__token2id[token]

        except Exception as err:
            print(err)

    def save(self, outputPath):
        """Saves the lexic model."""

        with open(outputPath, 'w', encoding='utf-8') as f:
            dumpDict = {'corpusTotalTokens': self.__corpusTotalTokens,
                'tokenFreq': self.__vocabulary,
                'token2id': self.__token2id}
            f.write(json.dumps(dumpDict))

    def load(self, inputPath):
        """Loads a lexic model."""
        
        with open(inputPath, encoding='utf-8') as f:
            dumpDict = json.loads(f.read())
            self.__corpusTotalTokens = dumpDict['corpusTotalTokens']
            self.__vocabulary = dumpDict['tokenFreq']
            self.__token2id = dumpDict['token2id']