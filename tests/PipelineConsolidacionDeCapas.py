import sys
sys.path.append('..')
from spectraltrep.layerConsolidation import Projector


projector = Projector("../data/SalidaAssembler.jsonl")
path = "../data/data_dummy_spectre_"
projector.resamble(path+"lexic.jsonl", path+"syntactic.jsonl", path+"semantic.jsonl")