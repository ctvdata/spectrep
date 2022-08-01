import sys
sys.path.append('..')
from spectraltrep.featureExtraction import VectorizerFactory, Doc2VecCorpusReader
from spectraltrep.utils import DocumentSink
import time

if __name__ == '__main__':
    init = time.time()

    print('Initializing ppeline objects')
    cr = Doc2VecCorpusReader('../data/pan_uniquedocs_short.jsonl')
    vw = DocumentSink('./outputs/SemanticVectorsTrain.jsonl', False)
    vf = VectorizerFactory()
    sv = vf.createSemanticVectorizer(vw, cr, 3)
    
    print('Training model')
    sv.fit()
    
    print('Performing vectorization')
    sv.transform()

    print('Save results')
    sv.saveModel('./outputs/dv2Model')
    vw.saveCorpus()

    print('Process finished in %.02f seconds' % (time.time()-init))