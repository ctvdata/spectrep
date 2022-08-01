import sys
sys.path.append('..')
from spectraltrep.featureExtraction import VectorizerFactory
from spectraltrep.utils import CorpusReader, DocumentSink
import time

if __name__ == '__main__':
    init = time.time()

    print('Initializing ppeline objects')
    cr = CorpusReader('../data/pan_uniquedocs_short.jsonl', 100)
    vw = DocumentSink('./outputs/LexicVectors.jsonl', False)
    vf = VectorizerFactory()
    lv = vf.createLexicVectorizer(vw, cr)
    
    print('Training model')
    lv.fit()
    
    print('Performing vectorization')
    lv.transform()

    print('Save results')
    lv.saveModel('outputs/lexicModel.json')
    vw.saveCorpus()

    print('Process finished in %.02f seconds' % (time.time()-init))