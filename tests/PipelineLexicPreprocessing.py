import sys
sys.path.append('..')
import numpy as np
from spectraltrep.preProcessing import PreprocessorFactory
from spectraltrep.utils import DocumentSink, CorpusReader, LockedIterator
import time

init = time.time()
numThreads = 1 # Number of preprocessing threads
batchSize = 3000 # Number of documents delivered by Corpus Reader per batch

# We create the factory of preprocessors, the corpus reader, and the sink that receives the preprocessed documents
ppf = PreprocessorFactory()
cr = CorpusReader('../data/data.jsonl', batchSize).getBatch()
lockedCr = LockedIterator(cr)
ds = DocumentSink("./outputs/SalidaPipelinePreProcesamientoLexico.jsonl", True)

# We initialize preprocessing threads
lexicPreprocessingThreads = []
for _ in np.arange(numThreads):
    lpp = ppf.createLexicPreprocessor(lockedCr, ds)
    lpp.start()
    lexicPreprocessingThreads.append(lpp)

# Wait for the preprocessing threads to finish
for t in lexicPreprocessingThreads:
    t.join()

# We save the preprocessed corpus
ds.saveCorpus()
print('%d threads, %.02f seconds' % (numThreads,time.time() - init))