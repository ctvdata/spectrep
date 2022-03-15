import sys
sys.path.append('..')
from spectraltrep.preProcessing import PreProcessingFacade
import time

def main():
    inputPath = '../data/data_sample.jsonl' # Archivo de entrada
    outputPath = '../data/SalidaPipelinePreProcesamiento.jsonl' # Archivo de salida
    preProcessingType = ['lex','syn','sem'] # Tipo de preprocesamiento aplicable ['lex', 'syn', 'sem']
    numThreads = 1 # Numero de hilos de preprocesamiento
    batchSize = 10 # Numero de documentos que entrega CorpusReader por batch
    sortedOutput = True

    init = time.time()
    ppf = PreProcessingFacade()
    ppf.preProcess(inputPath, outputPath, preProcessingType, numThreads, batchSize, sortedOutput)
    print('%d hilos, %.02f segundos' % (numThreads,time.time() - init))

if __name__ == '__main__':
    main()