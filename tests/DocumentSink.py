import sys
import numpy as np
sys.path.append('..')
from spectraltrep.preProcessing import DocumentSink

sinkOrder = DocumentSink("../data/SalidaDocumentSinkOrdenado.jsonl", True)
sink = DocumentSink("../data/SalidaDocumentSink.jsonl", False)
sinkVector = DocumentSink("../data/SalidaVectorWritter.jsonl", False)

arr = np.arange(10)
np.random.shuffle(arr)
print(arr)
for i in arr:
    batch = (i,[{'id': int(3*i + 1), 'text': 'contenido {}'.format(3*i + 1)},
                {'id': int(3*i + 2), 'text': 'contenido {}'.format(3*i + 2)},
                {'id': int(3*i + 3), 'text': 'contenido {}'.format(3*i + 3)}])
    sinkOrder.addPreprocessedBatch(batch)
    sink.addPreprocessedBatch(batch)
    sinkVector.addPreprocessedBatch((i,[{'id': int(3*i + 1), 'vector': np.arange(10)},
                                        {'id': int(3*i + 2), 'vector': np.arange(10)},
                                        {'id': int(3*i + 3), 'vector': np.arange(10)}]))
    
sinkOrder.saveCorpus()
sink.saveCorpus()
sinkVector.saveCorpus()