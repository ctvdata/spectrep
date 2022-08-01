import sys
sys.path.append('..')
import numpy as np
from spectraltrep.preProcessing import CorpusReader

# We create the reader object 
docReader = CorpusReader('../data/data.jsonl')
#docReader = CorpusReader('../data/SalidaDocumentSinkOrdenado.jsonl')
#docReader = CorpusReader('../data/SalidaVectorWritter.jsonl')

# We invoke the get batch method 10 times
for _ in np.arange(10):
    print(docReader.getBatch())
    print("\n")