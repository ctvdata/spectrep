import sys
import numpy as np
sys.path.append('..')
from spectraltrep.preProcessing import DocumentSink

sink = DocumentSink()

arr = np.arange(10)
np.random.shuffle(arr)
print(arr)
for i in arr:
    sink.addPreprocessedBatch({'batchId': i,
                                'content': [{'id': 3*i + 1, 'text': 'contenido {}'.format(3*i + 1)},
                                            {'id': 3*i + 2, 'text': 'contenido {}'.format(3*i + 2)},
                                            {'id': 3*i + 3, 'text': 'contenido {}'.format(3*i + 3)}]})
    
sink.saveCorpus("../data/SalidaDocumentSink.txt")