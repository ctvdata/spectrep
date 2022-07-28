import sys
sys.path.append('..')
from spectraltrep.layerConsolidation import SpectraAssembler


assembler = SpectraAssembler("./outputs/SalidaAssembler.jsonl")
path = "../data/data_dummy_spectre_"
assembler.assemble(path+"lexic.jsonl", path+"syntactic.jsonl", path+"semantic.jsonl")