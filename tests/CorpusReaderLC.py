import sys
sys.path.append('..')
import numpy as np
from spectraltrep.layerConsolidation import CorpusReader

# Creamos el objeto reader 
docReader = CorpusReader()
# Creamos el generador.
spectre_gen = docReader.read_spectre('../data/data_dummy_spectre_lexic.jsonl')

#Invocamos 10 veces el generador read_spectre.
for _ in np.arange(102):
    print(next(spectre_gen, '<EOS>'))
    print("\n")