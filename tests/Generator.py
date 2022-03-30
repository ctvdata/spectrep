import json
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.tokenize import word_tokenize
import pdb

class MyCorpus:
    """
    Clase generator para pasar documentos a doc2vec gensim
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
    # Creamos el generador de documentos
    corpus = MyCorpus('../data\pan_uniquedocs_short.jsonl')
    i = 0
    for item in corpus:
        # print(item[1][0])
        if i > 3:
            break
        print(item)
        i+=1
