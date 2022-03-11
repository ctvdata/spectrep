import sys
sys.path.append('..')
import numpy as np
from spectraltrep.preProcessing import PreprocessorFactory, CorpusReader, DocumentSink
import time

init = time.time()
numThreads = 2

# Creamos la fabrica de preprocesadores, el lector de corpus y el sink que recibe los documentos preprocesados
ppf = PreprocessorFactory()
cr = CorpusReader('../data/data.jsonl')
ds = DocumentSink()

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
ds.saveCorpus('../data/SalidaPipelinePreProcesamientoLexico.jsonl')
print('%d hilos, %.02f segundos' % (numThreads,time.time() - init))