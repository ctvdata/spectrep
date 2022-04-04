import sys
sys.path.append("..")
from spectraltrep.spaceUnification import CorpusReader
cr = CorpusReader('.\outputs\SemanticVectors2.jsonl')
print(f"{cr.numLines} lineas")

print("Leyendo vectores de entrenamieto")
gen = cr.readTrainingFeatureVectors(20)
for idx,item in enumerate(gen):
    print(idx, item)

print("\nLeyendo vectores para transformacion")
gen = cr.readFeatureVectors()
for item in gen:
    print(item)

print("leyendo vectores full")
print(cr.readFeatureVectorsFull())