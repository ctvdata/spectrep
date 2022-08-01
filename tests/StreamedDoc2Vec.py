from smart_open import open  # for transparently opening remote files
import json
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.tokenize import word_tokenize
import pdb

"""
Doc2Vec training test using a document stream method so 
as not to load the complete corpus in memory.
"""
class MyCorpus:
    """
    Generator class to pass documents to doc2vec gensim
    """
    def __init__(self, inputPath):
            self.__inputPath = inputPath

    def __iter__(self):        
        with open(self.__inputPath) as f:
            for line in f:
                # pdb.set_trace()
                line = json.loads(line)
                tokens = word_tokenize(line['text'])
                # assume there's one document per line, tokens separated by whitespace
                yield TaggedDocument(tokens, [int(line['id'])])

if __name__ == '__main__':
    # We create the document generator
    corpus = MyCorpus('../data\pan_uniquedocs_short.jsonl')
    

    # We create and train the doc2vec model
    model = Doc2Vec(vector_size=5, min_count=2, epochs=1)
    print('Building vocabulary')
    model.build_vocab(corpus)
    print('Training model')
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    pdb.set_trace()