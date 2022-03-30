from abc import ABCMeta, abstractmethod
from msilib.schema import Property
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json
from nltk.tokenize import word_tokenize
import json
import numpy as np

from spectraltrep.layerConsolidation import CorpusReader

class Reader(metaclass=ABCMeta):
    @abstractmethod
    def readCorpus(self, inputPath):
        pass

class Doc2VecCorpusReader(Reader):
    def __init__(self, inputPath):
            self.__inputPath = inputPath

    def __iter__(self):        
        with open(self.__inputPath) as f:
            for line in f:
                line = json.loads(line)
                tokens = word_tokenize(line['text'])
                yield TaggedDocument(tokens, [int(line['id'])])

    def readCorpus(self, inputPath):
        pass

class Vectorizer(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, corpus):
        pass

    @abstractmethod
    def transform(self, corpus):
        pass

    @abstractmethod
    def saveModel(self, outputPath):
        pass

class LexicVectorizer(Vectorizer):
    def __init__(self, vectorWriter, corpusReader=None):
        self.__model = LexicModel()
        self.__vectorWriter = vectorWriter
        self.__corpusReader = corpusReader

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, inputPath):
        model = LexicModel()
        model.load(inputPath)
        self.__model = model

    def fit(self):
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
        self.__model.save(outputPath)

class SyntacticVectorizer(Vectorizer):
    def fit(self, corpus):
        pass

    def transform(self, corpus):
        pass

    def saveModel(self, outputPath):
        pass

class SemanticVectorizer(Vectorizer):
    def __init__(self, vectorWriter, corpusReader=None, vectorSize=50, minCount=1, epochs=10):
        self.__model = Doc2Vec(vector_size=vectorSize, min_count=minCount, epochs=epochs)
        self.__vectorWriter = vectorWriter
        self.__corpusReader = corpusReader

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, inputPath):
        self.__model = Doc2Vec.load(inputPath)
    
    def fit(self):
        try:
            if self.__corpusReader is None:
                raise Exception('No se ha definido un corpus reader en el constructor para el entrenamiento.')
            else:
                self.__model.build_vocab(self.__corpusReader)
                self.__model.train(self.__corpusReader, total_examples=self.__model.corpus_count, epochs=self.__model.epochs)            
        except Exception as err:
            print(err)        

    def transform(self, corpusReader=None):
        cr = self.__corpusReader if corpusReader is None else corpusReader

        for doc in cr:
            idDoc = doc[1][0]
            docVec = self.__model.dv[idDoc]
            batch = (idDoc,[{'id':idDoc, 'vector':docVec}])
            self.__vectorWriter.addPreprocessedBatch(batch)        

    def saveModel(self, outputPath):
        self.__model.save(outputPath)

class VectorizerAbstractFactory(metaclass=ABCMeta):
    @abstractmethod
    def createLexicVectorizer(self):
        pass

    @abstractmethod
    def createSyntacticVectorizer(self):
        pass

    @abstractmethod
    def createSemanticVectorizer(self):
        pass

class VectorizerFactory(VectorizerAbstractFactory):
    def createLexicVectorizer(self, vectorWriter, corpusReader=None):
        return LexicVectorizer(vectorWriter, corpusReader)

    def createSyntacticVectorizer(self, vectorWriter, corpusReader=None, vectorSize=50, minCount=1, epochs=10):
        return SemanticVectorizer(vectorWriter, corpusReader, vectorSize, minCount, epochs)

        
    def createSemanticVectorizer(self, vectorWriter, corpusReader=None, vectorSize=50, minCount=1, epochs=10):
        return SemanticVectorizer(vectorWriter, corpusReader, vectorSize, minCount, epochs)

class LexicModel():
    def __init__(self):
        self.__vocabulary = dict()
        self.__corpusTotalTokens = 0

    @property
    def corpusTotalTokens(self):
        return self.__corpusTotalTokens

    @property
    def vocabularyLength(self):
        return len(self.__token2id.keys())

    def addToken(self, token):
        if token not in self.__vocabulary.keys():
            self.__vocabulary[token] = 1
            self.__corpusTotalTokens += 1
        else:
            self.__vocabulary[token] += 1
            self.__corpusTotalTokens += 1

    def getTokenProbability(self, token):
        if token in self.__vocabulary.keys():
            return self.__vocabulary[token] / self.__corpusTotalTokens
        else:
            return 0

    def getTokenFrequency(self, token):
        if token in self.__vocabulary.keys():
            return self.__vocabulary[token]
        else:
            return 0

    def getVocabulary(self):
        return self.__vocabulary.keys()

    def setToken2Id(self):
        self.__token2id = dict()
        for idx, token in enumerate(self.__vocabulary.keys()):
            self.__token2id[token] = idx
    
    def getTokenId(self, token):
        try:
            if token not in self.__token2id.keys():
                raise Exception("No existe el token en el vocabulario")
            else:
                return self.__token2id[token]

        except Exception as err:
            print(err)

    def save(self, outputPath):
        with open(outputPath, 'w', encoding='utf-8') as f:
            dumpDict = {'corpusTotalTokens': self.__corpusTotalTokens,
                'tokenFreq': self.__vocabulary,
                'token2id': self.__token2id}
            f.write(json.dumps(dumpDict))

    def load(self, inputPath):
        with open(inputPath, encoding='utf-8') as f:
            dumpDict = json.loads(f.read())
            self.__corpusTotalTokens = dumpDict['corpusTotalTokens']
            self.__vocabulary = dumpDict['tokenFreq']
            self.__token2id = dumpDict['token2id']