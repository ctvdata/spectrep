from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import sys
sys.path.append('..')
from minisom.minisom import MiniSom
import json
import pickle

class Reader(metaclass=ABCMeta):
    """
    Lector de vectores de características.
    """
    @abstractmethod
    def readFeatureVectors(self):
        """
        Carga vectores de características en memoria.
        """
        pass

class VectorReader(Reader):
    """
    Lector de vectores de características.

    Attributes:
        inputPath (str): Ruta del archivo de entrada. Este archivo debe ser un jsonl
        con el siguiente formato:

            {"id": idDelDocumento, "vector": vectorDeCaracterísticas}
            ...

        numLines (int): Número de líneas en el archivo de entrada.
    """

    def __init__(self, inputPath):
        """
        Inicializa el lector de vectores de características.

        Args:
            inputPath (str): Ruta del archivo de entrada. Este archivo debe ser un jsonl
            con el siguiente formato:

            {"id": idDelDocumento, "vector": vectorDeCaracterísticas}
            ...
        """
        self.__inputPath = inputPath
        self.__numLines = 0

        with open(self.__inputPath) as f:
            for _ in f:
                self.__numLines +=1

    def readFeatureVector(self):
        """
        Generador que carga vectores de características en memoria.
        Esta función no está en uso para esta versión de la api.

        Yields:
            Vector de características.
        """

        with open(self.__inputPath) as f:
                for line in f:
                    vector = json.loads(line)
                    yield vector

    #Hay que buscar si es posibe acopar esta funcion a SimpSom
    # def readTrainingFeatureVectors(self, size):
    #     indices = np.random.randint(0, self.__numLines, size)
    #     for idx in indices:
    #         with open(self.__inputPath) as f:
    #             for line in f:
    #                 vector = json.loads(line)
    #                 if vector['id'] == idx:
    #                     yield np.array(vector['vector'])
    #                     break

    def readFeatureVectors(self):
        """
        Carga vectores de características en memoria.

        Returns:
            Conjunto de vectores de características (NumPy.ndarray).
        """
        vectors = list()
        with open(self.__inputPath) as f:
            for line in f:
                vector = json.loads(line)
                vectors.append(vector['vector'])
        return np.array(vectors)

class Projector:
    """
    Proyector de vectores de características.
    
    Hace uso de un modelo de Mapas Auto-organizados (SOM) para obtener espectros
    del contenido.

    Attributes:
        netLength (int): Número de dimensiones del espectro a obtener (netLength * netLength).

        model (MiniSOM): Modelo de mapas auto-organizados.
    """

    def __init__(self, netLength, numDimensions, learningRate=0.01):
        """
        Inicializa el objeto Projector
        
        Args:
            netLength (int): Número de dimensiones del espectro a obtener (netLength * netLength).

            numDimensions (int): Número de dimensiones de los vectores de entrada.

            learningRate (float): Taza de aprendizaje para el SOM.
        """

        self.__netLength=netLength
        self.__model = MiniSom(netLength, netLength, numDimensions, learning_rate=learningRate)

    def fit(self, featureVectors, epochs=10):
        """
        Realiza el entrenamiento del modelo SOM para obtener los espectros
        de contenido.

        Args:
            featureVectors (numpy.ndarray): Conjunto de vectores de características
            para realizar el entrenamiento.

            epochs (int): Número de iteraciones en las que se realizará el entrenamiento.
        """
        print("Inicializando pesos")
        # self.__model.random_weights_init(featureVectors)
        self.__model.pca_weights_init(featureVectors)
        print("Inicia el entrenamiento de som")
        # self.__model.get_weights().dump("preTrainWeights.npy")
        self.__model.train(featureVectors, epochs, verbose=True)
        # self.__model.get_weights().dump("postTrainWeights.npy")

    def getProjection(self, featureVectors, documentSink=None):
        """
        Obteniene las proyecciones de un conjunto de vectores de características
        en el modelo SOM previamente entrenado.

        Args:
            featureVectors (numpy.ndarray): Conjunto de vectores de características
            a proyectar.

            documentSink (DocumentSink): Recolector de vectores proyectados. En caso de
            que no se proveea uno se devuelve una matriz de proyecciones.
        
        Returns:
            Matriz de proyecciones en caso de que no se proveea un reclector
            de vectores proyectados (DocumentSink).
        """
        nrows = featureVectors.shape[0]
        i=0
        result_matrix = []
        nodes = self.__model.get_weights()
        for row in featureVectors:
            image_document = np.zeros(shape=(self.__netLength,self.__netLength), dtype=np.float32)
        
            for i in np.arange(image_document.shape[0]):
                for j in np.arange(image_document.shape[1]):
                    image_document[i,j] = 1/np.linalg.norm(row-nodes[i,j])
            
            result_matrix.append(image_document)
            s = "{}% Complete".format(int((i*100)/nrows))
            print(s, end='\r')
            i=i+1
        
        result_matrix = np.array(result_matrix)
        # result_matrix = 255.9 * (result_matrix - result_matrix.min()) / (result_matrix.max() - result_matrix.min())
        result_matrix = (result_matrix - result_matrix.min()) / (result_matrix.max() - result_matrix.min()) #Solamente valores en el rango [0,1]

        print("100% Complete", end='\r')
        
        if documentSink is None:
            return result_matrix
        else:
            spectra = []
            for idx, spectre in enumerate(result_matrix[:]):
                spectra.append({'id':idx, 'spectre':spectre})    
            documentSink.addPreprocessedBatch((0,spectra))
            


        # for row in featureVectors:
        #     j=0
        #     image_document=np.zeros(shape=self.__netLength*self.__netLength, dtype=np.float32)
        #     for node in self.net.nodeList:
        #         image_document[j] = 1/np.linalg.norm(row-node.weights)
        #         j=j+1
        #     mn=image_document.min()
        #     mx=image_document.max()
        #     image_document = (255.9 * (image_document-mn)/(mx-mn)).astype(np.uint8)
        #     image_document=image_document.reshape(self.netLength,self.netLength)
        #     result_matrix.append(image_document)  
        #     s = "{}% Complete".format(int((i*100)/nrows))
        #     print(s, end='\r')
        #     i=i+1
        # print("100% Complete", end='\r')
        # return result_matrix 

    def saveSomModel(self, outputPath):
        """
        Guarda el modelo SOM entrenado.

        Args:
            outputPath (str): Ruta de salida para el archivo del modelo de SOM.
        """
        with open(outputPath, 'wb') as outfile:
            pickle.dump(self.__model, outfile)

    def loadSomModel(self, inputPath):
        """
        Carga un modelo de SOM previamente entrenado.

        Args: 
            inputPath (str): Ruta del archivo de entrada.
        """
        with open(inputPath, 'rb') as infile:
            self.__model = pickle.load(infile)
            self.__netLength = self.__model.get_weights().shape[0]