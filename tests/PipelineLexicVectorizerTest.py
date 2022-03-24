import sys
sys.path.append('..')
from spectraltrep.featureExtraction import VectorizerFactory
from spectraltrep.preProcessing import CorpusReader, DocumentSink, LockedIterator
import time

if __name__ == '__main__':
    init = time.time()

    print('Inicializando objetos del pipeline')
    cr = CorpusReader('../data/pan_uniquedocs_short_test.jsonl', 1)
    vw = DocumentSink('outputs/LexicVectorsTest.jsonl', False)
    vf = VectorizerFactory()
    lv = vf.createLexicVectorizer(vw)
    lv.model = 'outputs\lexicModel.json'
    
    print('Realizando vectorizacion')
    lv.transform(cr)

    print('Guardando resultados')
    vw.saveCorpus()

    print('Proceso finalizado en %.02f segundos' % (time.time()-init))