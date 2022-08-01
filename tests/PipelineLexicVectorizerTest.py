import sys
sys.path.append('..')
from spectraltrep.featureExtraction import VectorizerFactory
from spectraltrep.utils import CorpusReader, DocumentSink
import time

if __name__ == '__main__':
    init = time.time()

    print('Initializing ppeline objects')
    cr = CorpusReader('../data/pan_uniquedocs_short_test.jsonl', 1)
    vw = DocumentSink('outputs/LexicVectorsTest.jsonl', False)
    vf = VectorizerFactory()
    lv = vf.createLexicVectorizer(vw)
    lv.model = 'outputs\lexicModel.json'
    
    print('Performing vectorization')
    lv.transform(cr)

    print('Save results')
    vw.saveCorpus()

    print('Process finished in %.02f seconds' % (time.time()-init))