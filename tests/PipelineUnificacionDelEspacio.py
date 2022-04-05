import sys
sys.path.append('..')
from spectraltrep.spaceUnification import CorpusReader, Projector
from spectraltrep.preProcessing import DocumentSink

cr = CorpusReader('./outputs/SemanticVectors2.jsonl')
proj = Projector(3,3)
sink = DocumentSink('./outputs/SemanticSpectra.jsonl', False)

print("Leyendo vectores")
data = cr.readFeatureVectors()
print("Entrenando som")
proj.fit(data,10)
# print("Obteninendo espectros")
# proj.getProjection(data, sink)
print('Guardando modelo')
proj.saveSomModel('./outputs/model.som')

print('Cargando modelo')
proj = Projector(0,0)
proj.loadSomModel('./outputs/model.som')
# print(proj.netLenght)
print("Obteninendo espectros")
proj.getProjection(data, sink)