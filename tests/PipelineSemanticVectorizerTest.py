import sys
sys.path.append('..')
from spectraltrep.featureExtraction import VectorizerFactory, Doc2VecCorpusReader
from spectraltrep.preProcessing import DocumentSink
import time

if __name__ == '__main__':
    init = time.time()

    print('Inicializando objetos del pipeline')
    vw = DocumentSink('outputs/SemanticVectorsTest.jsonl', False)
    vf = VectorizerFactory()
    sv = vf.createSemanticVectorizer(vw)
    sv.model = 'outputs/dv2Model'
    
    print('Realizando vectorizacion')
    cr = Doc2VecCorpusReader('../data/pan_uniquedocs_short_test.jsonl')
    sv.transform(cr)

    print('Guardando resultados')
    vw.saveCorpus()

    print('Proceso finalizado en %.02f segundos' % (time.time()-init))