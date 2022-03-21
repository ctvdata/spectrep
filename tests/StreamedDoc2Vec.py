from smart_open import open  # for transparently opening remote files
import json
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.tokenize import word_tokenize
import pdb

"""
Prueba de entrenamiento de Doc2Vec usando un metodo de stream de documentos
para no cargar el corpus completo en memoria.
"""
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
    

    # Creamos y entrenamos el modelo doc2vec
    model = Doc2Vec(vector_size=5, min_count=2, epochs=1)
    print('Building vocabulary')
    model.build_vocab(corpus)
    print('Training model')
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    # doc = 0
    # print(f'Vector of document {doc}')
    # print(model.dv[doc])
    pdb.set_trace()

    # newdoc = 'hola mi nombre es mel'
    # print("Vector inferido de: " + newdoc)
    # print(model.infer_vector(newdoc.split()))