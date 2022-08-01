import sys
sys.path.append('..')
from spectraltrep.featureExtraction import VectorizerFactory, Doc2VecCorpusReader
from spectraltrep.utils import DocumentSink
import time

if __name__ == '__main__':
    init = time.time()

    print('Initializing ppeline objects')
    vw = DocumentSink('./outputs/SemanticVectorsTest.jsonl', False)
    vf = VectorizerFactory()
    sv = vf.createSemanticVectorizer(vw)
    sv.model = './outputs/dv2Model'
    
    print('Performing vectorization')
    cr = Doc2VecCorpusReader('../data/pan_uniquedocs_short_test.jsonl')
    sv.transform(cr)

    print('Save results')
    vw.saveCorpus()

    print('Process finished in %.02f seconds' % (time.time()-init))