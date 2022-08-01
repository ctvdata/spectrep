import sys
sys.path.append('..')
import numpy as np
from spectraltrep.layerConsolidation import SpectraReader

# We create the reader object
docReader = SpectraReader('../data/data_dummy_spectre_lexic.jsonl')
# We create the generator.
spectre_gen = docReader.readSpectra()

# We call the read_spectre generator 10 times.
for _ in np.arange(102):
    print(next(spectre_gen, '<EOS>'))
    print("\n")