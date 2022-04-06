import sys
sys.path.append('..')
from spectraltrep.spaceUnification import VectorReader, Projector
from spectraltrep.utils import DocumentSink

# vr = VectorReader('./outputs/SemanticVectorsTrain.jsonl')
vr = VectorReader('./outputs/LexicVectors.jsonl')
proj = Projector(3,21743) # Indicamos el tamano de la capa de salida y las dimensiones de entrada
sink = DocumentSink('./outputs/LexicSpectra.jsonl', False)
# sink = DocumentSink('./outputs/SemanticSpectra.jsonl', False)

print("Leyendo vectores")
data = vr.readFeatureVectors()
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