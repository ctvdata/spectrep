import sys
sys.path.append('..')
from spectraltrep.featureExtraction import VectorizerFactory, Doc2VecCorpusReader
from spectraltrep.utils import DocumentSink
import time

if __name__ == '__main__':
    init = time.time()

    print('Inicializando objetos del pipeline')
    cr = Doc2VecCorpusReader('../data/pan_uniquedocs_short.jsonl')
    vw = DocumentSink('./outputs/SemanticVectorsTrain.jsonl', False)
    vf = VectorizerFactory()
    sv = vf.createSemanticVectorizer(vw, cr, 3)
    
    print('Entrenando modelo')
    sv.fit()
    
    print('Realizando vectorizacion')
    sv.transform()

    print('Guardando resultados')
    sv.saveModel('./outputs/dv2Model')
    vw.saveCorpus()

    print('Proceso finalizado en %.02f segundos' % (time.time()-init))