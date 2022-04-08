import sys
sys.path.append('..')
import numpy as np
from spectraltrep.layerConsolidation import SpectraReader

# Creamos el objeto reader 
docReader = SpectraReader('../data/data_dummy_spectre_lexic.jsonl')
# Creamos el generador.
spectre_gen = docReader.readSpectra()

#Invocamos 10 veces el generador read_spectre.
for _ in np.arange(102):
    print(next(spectre_gen, '<EOS>'))
    print("\n")