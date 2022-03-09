import sys
sys.path.append('..')
import numpy as np
from spectraltrep.preProcessing import CorpusReader

# Creamos el objeto reader 
docReader = CorpusReader('../data/data.jsonl')

#Invocamos 10 veces el metodo get batch
for _ in np.arange(10):
    print(docReader.getBatch())
    print("\n")