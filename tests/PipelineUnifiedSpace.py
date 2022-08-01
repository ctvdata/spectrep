import sys
sys.path.append('..')
from spectraltrep.spaceUnification import VectorReader, Projector
from spectraltrep.utils import DocumentSink

# vr = VectorReader('./outputs/SemanticVectorsTrain.jsonl')
vr = VectorReader('./outputs/LexicVectors.jsonl')
proj = Projector(3,21743) # Indicamos el tamano de la capa de salida y las dimensiones de entrada
sink = DocumentSink('./outputs/LexicSpectra.jsonl', False)
# sink = DocumentSink('./outputs/SemanticSpectra.jsonl', False)

print("Reading vectors")
data = vr.readFeatureVectors()
print("Training SOM")
proj.fit(data,10)
print('Guardando modelo')
proj.saveSomModel('./outputs/model.som')

print('Save model')
proj = Projector(0,0)
proj.loadSomModel('./outputs/model.som')
print("Obtaining spectra")
proj.getProjection(data, sink)