import sys
sys.path.append('..')
import numpy as np
from spectraltrep.preProcessing import PreprocessorFactory, CorpusReader, DocumentSink
from threading import Lock
import time

init = time.time()
numThreads = 1 # Numero de hilos de preprocesamiento
batchSize = 3000 # Numero de documentos que entrega CorpusReader por batch

# Creamos la fabrica de preprocesadores, el lector de corpus y el sink que recibe los documentos preprocesados
ppf = PreprocessorFactory()
cr = CorpusReader('../data/data.jsonl', batchSize)
ds = DocumentSink(True, "../data/SalidaPipelinePreProcesamientoLexico.jsonl")

# Inicializamos hilos de preprocesamiento
lexicPreprocessingThreads = []
for _ in np.arange(numThreads):
    lpp = ppf.createLexicPreprocessor(cr, ds)
    lpp.start()
    lexicPreprocessingThreads.append(lpp)

# Esperamos a que terminen los hilos de preprocesamiento
for t in lexicPreprocessingThreads:
    t.join()

# Guardamos el corpus preprocesado
ds.saveCorpus()
print('%d hilos, %.02f segundos' % (numThreads,time.time() - init))