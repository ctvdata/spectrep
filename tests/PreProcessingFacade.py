import sys
sys.path.append('..')
from spectraltrep.preProcessing import PreProcessingFacade
import time

inputPath = '../data/data.jsonl' # Archivo de entrada
outputPath = '../data/SalidaPipelinePreProcesamiento.jsonl' # Archivo de salida
preProcessingType = ['lex','syn','sem'] # Tipo de preprocesamiento aplicable ['lex', 'syn', 'sem']
numThreads = 2 # Numero de hilos de preprocesamiento
batchSize = 3000 # Numero de documentos que entrega CorpusReader por batch

init = time.time()
ppf = PreProcessingFacade()
ppf.preProcess(inputPath, outputPath, preProcessingType, numThreads, batchSize)
print('%d hilos, %.02f segundos' % (numThreads,time.time() - init))