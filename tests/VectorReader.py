import sys
sys.path.append("..")
from spectraltrep.spaceUnification import CorpusReader
cr = CorpusReader('.\outputs\SemanticVectors2.jsonl')
print(f"{cr.numLines} lineas")

print("Reading training vectors")
gen = cr.readTrainingFeatureVectors(20)
for idx,item in enumerate(gen):
    print(idx, item)

print("\nReading vectors for transformation")
gen = cr.readFeatureVectors()
for item in gen:
    print(item)

print("Reading Full vectors")
print(cr.readFeatureVectorsFull())