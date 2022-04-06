import sys
sys.path.append('..')
from spectraltrep.featureExtraction import VectorizerFactory
from spectraltrep.preProcessing import CorpusReader, DocumentSink
import time

if __name__ == '__main__':
    init = time.time()

    print('Inicializando objetos del pipeline')
    cr = CorpusReader('../data/pan_uniquedocs_short.jsonl', 100)
    vw = DocumentSink('outputs/LexicVectors.jsonl', False)
    vf = VectorizerFactory()
    lv = vf.createLexicVectorizer(vw, cr)
    
    print('Entrenando modelo')
    lv.fit()
    
    print('Realizando vectorizacion')
    lv.transform()

    print('Guardando resultados')
    lv.saveModel('outputs/lexicModel.json')
    vw.saveCorpus()

    print('Proceso finalizado en %.02f segundos' % (time.time()-init))